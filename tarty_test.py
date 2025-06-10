#!/usr/bin/env python3

import json
import subprocess
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from tarty import (
    DEFAULT_UPDATE_HOUR,
    ORCHESTRATION_CYCLE_INTERVAL,
    RUNNER_CONFIG_TIMEOUT,
    RUNNER_START_DELAY,
    VM_START_TIMEOUT,
    ConfigError,
    GitHubRunnerManager,
    ImageManager,
    RunnerConfig,
    SSHClient,
    TartVM,
    TartyConfig,
    TartyOrchestrator,
    VMState,
    run_ssh_command,
    run_tart_command,
)


def test_runner_config_valid():
    config_data = {
        'github_pat': 'test_pat',
        'organization': 'test_org',
        'repository': 'test_repo',
        'base_image': 'test_base',
        'runner_image': 'test_runner',
        'ssh_username': 'test_user',
        'convert_command': 'test_cmd',
        'update_hour': 10,
        'max_vms': 1,
        'labels': ['macos', 'test'],
    }
    config = RunnerConfig(**config_data)
    assert config.github_pat == 'test_pat'
    assert config.update_hour == 10
    assert config.max_vms == 1
    assert config.labels == ['macos', 'test']


@pytest.mark.parametrize(
    'field, value',
    [
        ('github_pat', ''),
        ('organization', ''),
        ('base_image', ''),
        ('runner_image', ''),
    ],
)
def test_runner_config_required_fields_empty(field, value):
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'base_image': 'base',
        'runner_image': 'runner',
    }
    config_data[field] = value
    with pytest.raises(ValueError, match='Field cannot be empty'):
        RunnerConfig(**config_data)


@pytest.mark.parametrize('hour', [-1, 24])
def test_runner_config_invalid_update_hour(hour):
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'repository': 'repo',
        'base_image': 'base',
        'runner_image': 'runner',
        'update_hour': hour,
    }
    with pytest.raises(
        ValueError, match='update_hour must be between 0 and 23'
    ):
        RunnerConfig(**config_data)


@pytest.mark.parametrize('max_vms', [0, 3])
def test_runner_config_invalid_max_vms(max_vms):
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'repository': 'repo',
        'base_image': 'base',
        'runner_image': 'runner',
        'max_vms': max_vms,
    }
    with pytest.raises(ValueError, match='max_vms must be 1 or 2'):
        RunnerConfig(**config_data)


def test_runner_config_optional_repository():
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'base_image': 'base',
        'runner_image': 'runner',
    }
    config = RunnerConfig(**config_data)
    assert config.repository is None


def test_runner_config_optional_convert_command():
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'base_image': 'base',
        'runner_image': 'runner',
    }
    config = RunnerConfig(**config_data)
    assert config.convert_command is None


def test_runner_config_optional_labels():
    config_data = {
        'github_pat': 'pat',
        'organization': 'org',
        'base_image': 'base',
        'runner_image': 'runner',
    }
    config = RunnerConfig(**config_data)
    assert config.labels is None


@pytest.fixture
def mock_config_file(tmp_path):
    config_data = {
        'github_pat': 'mock_pat',
        'organization': 'mock_org',
        'repository': 'mock_repo',
        'base_image': 'mock_base',
        'runner_image': 'mock_runner',
        'ssh_username': 'mock_user',
        'convert_command': 'mock_cmd',
        'update_hour': 5,
        'max_vms': 2,
        'labels': ['macos', 'mock'],
    }
    config_path = tmp_path / 'test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path, config_data


def test_tarty_config_load_success(mock_config_file):
    config_path, expected_data = mock_config_file
    config = TartyConfig(str(config_path))
    assert config.github_pat == expected_data['github_pat']
    assert config.organization == expected_data['organization']
    assert config.repository == expected_data['repository']
    assert config.base_image == expected_data['base_image']
    assert config.runner_image == expected_data['runner_image']
    assert config.ssh_username == expected_data['ssh_username']
    assert config.convert_command == expected_data['convert_command']
    assert config.update_hour == expected_data['update_hour']
    assert config.max_vms == expected_data['max_vms']
    assert config.labels == expected_data['labels']


def test_tarty_config_file_not_found(tmp_path):
    non_existent_path = tmp_path / 'non_existent.json'
    with pytest.raises(ConfigError, match='Configuration file not found'):
        TartyConfig(str(non_existent_path))


def test_tarty_config_invalid_json(tmp_path):
    invalid_json_path = tmp_path / 'invalid.json'
    invalid_json_path.write_text('{ "key": "value" ')  # Malformed JSON
    with pytest.raises(ConfigError, match='Invalid JSON in config file'):
        TartyConfig(str(invalid_json_path))


def test_tarty_config_invalid_schema(tmp_path):
    invalid_schema_path = tmp_path / 'invalid_schema.json'
    invalid_schema_path.write_text(
        '{"github_pat": ""}'
    )  # Missing required fields
    with pytest.raises(ConfigError, match='Invalid configuration'):
        TartyConfig(str(invalid_schema_path))


@patch('subprocess.run')
def test_run_tart_command_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    result = run_tart_command(['list'])
    mock_run.assert_called_once_with(['tart', 'list'])
    assert result.returncode == 0


@patch('subprocess.run')
def test_run_ssh_command_success(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout='output', stderr=''
    )
    result = run_ssh_command('192.168.1.100', 'admin', 'ls')
    expected_cmd = [
        'ssh',
        '-o',
        'StrictHostKeyChecking=no',
        '-o',
        'UserKnownHostsFile=/dev/null',
        'admin@192.168.1.100',
        'ls',
    ]
    mock_run.assert_called_once_with(
        expected_cmd, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert result.stdout == 'output'


@patch('subprocess.run')
def test_run_ssh_command_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='error')
    result = run_ssh_command('192.168.1.100', 'admin', 'bad_cmd')
    assert result.returncode == 1
    assert result.stderr == 'error'


@patch('tarty.run_tart_command')
def test_cleanup_previous_tarty_vms_success(mock_run_tart_command):
    mock_run_tart_command.side_effect = [
        MagicMock(
            stdout='tarty-old-vm\nother-vm\ntarty-another\n', returncode=0
        ),
        MagicMock(returncode=0),
        MagicMock(returncode=0),
    ]

    from tarty import cleanup_previous_tarty_vms

    cleanup_previous_tarty_vms()

    mock_run_tart_command.assert_any_call(
        ['list', '-q'], capture_output=True, text=True, check=True
    )
    mock_run_tart_command.assert_any_call(
        ['delete', 'tarty-old-vm'], capture_output=True, check=True
    )
    mock_run_tart_command.assert_any_call(
        ['delete', 'tarty-another'], capture_output=True, check=True
    )
    assert mock_run_tart_command.call_count == 3


@patch('tarty.run_tart_command')
def test_cleanup_previous_tarty_vms_no_tarty_vms(mock_run_tart_command):
    mock_run_tart_command.return_value = MagicMock(
        stdout='other-vm\nanother-vm\n', returncode=0
    )

    from tarty import cleanup_previous_tarty_vms

    cleanup_previous_tarty_vms()

    mock_run_tart_command.assert_called_once_with(
        ['list', '-q'], capture_output=True, text=True, check=True
    )


@patch('tarty.run_tart_command')
def test_cleanup_previous_tarty_vms_list_fails(mock_run_tart_command):
    mock_run_tart_command.side_effect = subprocess.CalledProcessError(
        1, 'tart list'
    )

    from tarty import cleanup_previous_tarty_vms

    cleanup_previous_tarty_vms()

    mock_run_tart_command.assert_called_once_with(
        ['list', '-q'], capture_output=True, text=True, check=True
    )


@patch('tarty.run_tart_command')
def test_cleanup_previous_tarty_vms_delete_fails(mock_run_tart_command):
    mock_run_tart_command.side_effect = [
        MagicMock(stdout='tarty-old-vm\n', returncode=0),
        subprocess.CalledProcessError(1, 'tart delete'),
    ]

    from tarty import cleanup_previous_tarty_vms

    cleanup_previous_tarty_vms()

    mock_run_tart_command.assert_any_call(
        ['list', '-q'], capture_output=True, text=True, check=True
    )
    mock_run_tart_command.assert_any_call(
        ['delete', 'tarty-old-vm'], capture_output=True, check=True
    )


@patch('tarty.run_ssh_command')
def test_ssh_client_execute_command_success(mock_run_ssh_command):
    mock_run_ssh_command.return_value = MagicMock(returncode=0)
    client = SSHClient('192.168.1.100', 'admin')
    assert client.execute_command('test_cmd') is True
    mock_run_ssh_command.assert_called_once_with(
        '192.168.1.100', 'admin', 'test_cmd', 60
    )


@patch('tarty.run_ssh_command')
def test_ssh_client_execute_command_failure(mock_run_ssh_command):
    mock_run_ssh_command.return_value = MagicMock(
        returncode=1, stderr='ssh error'
    )
    client = SSHClient('192.168.1.100', 'admin')
    assert client.execute_command('test_cmd') is False


@patch('tarty.SSHClient')
@patch('tarty.run_tart_command')
@patch('tarty.threading.Thread')
def test_tart_vm_start_success(
    mock_thread, mock_run_tart_command, mock_ssh_client_cls
):
    mock_vm_ip = '192.168.1.101'
    mock_run_tart_command.return_value = MagicMock(
        stdout=mock_vm_ip, returncode=0
    )
    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.is_available.return_value = True

    vm = TartVM('test-vm', 'test-image')
    assert vm.start() is True
    assert vm.state == VMState.RUNNING
    assert vm._ip_address == mock_vm_ip
    mock_thread.assert_called_once()
    mock_ssh_client_instance.is_available.assert_called_once_with(
        timeout=VM_START_TIMEOUT
    )


@patch('tarty.SSHClient')
@patch('tarty.run_tart_command')
@patch('tarty.threading.Thread')
def test_tart_vm_start_fail_get_ip(
    mock_thread, mock_run_tart_command, mock_ssh_client_cls
):
    mock_run_tart_command.side_effect = [
        MagicMock(returncode=0),
        subprocess.CalledProcessError(
            1, ['tart', 'ip', 'test-vm'], stderr='no ip'
        ),
    ]
    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.is_available.return_value = False

    vm = TartVM('test-vm', 'test-image')
    assert vm.start() is False
    assert vm.state == VMState.STOPPED
    mock_thread.assert_called_once()
    mock_run_tart_command.assert_any_call(
        ['stop', 'test-vm'], check=True, capture_output=True
    )


@patch('tarty.SSHClient')
@patch('tarty.run_tart_command')
@patch('tarty.threading.Thread')
def test_tart_vm_start_fail_ssh_not_available(
    mock_thread, mock_run_tart_command, mock_ssh_client_cls
):
    mock_vm_ip = '192.168.1.101'
    mock_run_tart_command.return_value = MagicMock(
        stdout=mock_vm_ip, returncode=0
    )
    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.is_available.return_value = False

    vm = TartVM('test-vm', 'test-image')
    assert vm.start() is False
    assert vm.state == VMState.STOPPED
    mock_thread.assert_called_once()
    mock_ssh_client_instance.is_available.assert_called_once_with(
        timeout=VM_START_TIMEOUT
    )
    mock_run_tart_command.assert_any_call(
        ['stop', 'test-vm'], check=True, capture_output=True
    )


@patch('tarty.run_tart_command')
def test_tart_vm_stop_success(mock_run_tart_command):
    vm = TartVM('test-vm', 'test-image')
    vm.state = VMState.RUNNING  # Set to running to test stopping
    assert vm.stop() is True
    assert vm.state == VMState.STOPPED
    mock_run_tart_command.assert_called_once_with(
        ['stop', 'test-vm'], check=True, capture_output=True
    )


@patch('tarty.run_tart_command')
def test_tart_vm_stop_already_stopped(mock_run_tart_command):
    vm = TartVM('test-vm', 'test-image')
    vm.state = VMState.STOPPED
    assert vm.stop() is True
    mock_run_tart_command.assert_not_called()


@patch('tarty.run_tart_command')
def test_tart_vm_destroy_success(mock_run_tart_command):
    vm = TartVM('test-vm', 'test-image')
    vm.state = VMState.STOPPED
    assert vm.destroy() is True
    mock_run_tart_command.assert_called_once_with(
        ['delete', 'test-vm'], check=True, capture_output=True
    )


@patch('tarty.run_tart_command')
def test_tart_vm_destroy_stops_first(mock_run_tart_command):
    # Mock the tart get command for running check
    mock_run_tart_command.side_effect = [
        MagicMock(
            stdout='{"Running": true}', returncode=0
        ),  # For running check
        MagicMock(returncode=0),  # For stop command
        MagicMock(returncode=0),  # For delete command
    ]

    vm = TartVM('test-vm', 'test-image')
    vm.state = VMState.RUNNING
    assert vm.destroy() is True
    mock_run_tart_command.assert_any_call(
        ['get', 'test-vm', '--format', 'json'],
        capture_output=True,
        text=True,
        check=True,
    )
    mock_run_tart_command.assert_any_call(
        ['stop', 'test-vm'], check=True, capture_output=True
    )
    mock_run_tart_command.assert_any_call(
        ['delete', 'test-vm'], check=True, capture_output=True
    )


@pytest.fixture
def mock_runner_config():
    return MagicMock(
        github_pat='test_pat',
        organization='test_org',
        repository='test_repo',
        ssh_username='admin',
    )


@patch('urllib.request.urlopen')
@patch('json.loads')
def test_get_registration_token_success(
    mock_json_loads, mock_urlopen, mock_runner_config
):
    mock_json_loads.return_value = {'token': 'mock_token'}
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        b'{"token": "mock_token"}'
    )

    manager = GitHubRunnerManager(mock_runner_config)
    token = manager.get_registration_token()
    assert token == 'mock_token'
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == 'POST'
    assert 'Authorization' in req.headers
    assert 'Accept' in req.headers


@patch('urllib.request.urlopen', side_effect=Exception('API Error'))
def test_get_registration_token_failure(mock_urlopen, mock_runner_config):
    manager = GitHubRunnerManager(mock_runner_config)
    token = manager.get_registration_token()
    assert token is None


@patch(
    'tarty.GitHubRunnerManager.get_registration_token',
    return_value='mock_token',
)
@patch('tarty.SSHClient')
def test_register_runner_success(
    mock_ssh_client_cls, mock_get_token, mock_runner_config
):
    mock_vm = MagicMock()
    mock_vm.name = 'test-vm'
    mock_vm.get_vm_ip.return_value = '192.168.1.102'

    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.execute_command.side_effect = [
        True,
        True,
    ]

    with patch('tarty.RUNNER_START_DELAY', 0.1):
        manager = GitHubRunnerManager(mock_runner_config)
        assert manager.register_runner(mock_vm) is True
    assert mock_vm.runner_token == 'mock_token'
    mock_ssh_client_instance.execute_command.assert_any_call(
        'cd actions-runner && ./config.sh --url https://github.com/test_org/test_repo --token mock_token --name test-vm --ephemeral --unattended',
        RUNNER_CONFIG_TIMEOUT,
    )
    # Check that _start_runner_process was called instead of _start_runner
    assert mock_vm.runner_process is not None


@patch('tarty.GitHubRunnerManager.get_registration_token', return_value=None)
def test_register_runner_no_token(mock_get_token, mock_runner_config):
    mock_vm = MagicMock(name='test-vm')
    manager = GitHubRunnerManager(mock_runner_config)
    assert manager.register_runner(mock_vm) is False


@patch(
    'tarty.GitHubRunnerManager.get_registration_token',
    return_value='mock_token',
)
@patch('tarty.SSHClient')
def test_register_runner_config_fail(
    mock_ssh_client_cls, mock_get_token, mock_runner_config
):
    mock_vm = MagicMock(
        name='test-vm', get_vm_ip=MagicMock(return_value='192.168.1.102')
    )
    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.execute_command.return_value = False

    manager = GitHubRunnerManager(mock_runner_config)
    assert manager.register_runner(mock_vm) is False


@patch(
    'tarty.GitHubRunnerManager.get_registration_token',
    return_value='mock_token',
)
@patch('tarty.SSHClient')
@patch('tarty.threading.Thread')
@patch('time.sleep')
def test_register_runner_with_labels(
    mock_sleep,
    mock_thread,
    mock_ssh_client_cls,
    mock_get_token,
    mock_runner_config,
):
    mock_runner_config.labels = ['macos', 'tart']
    mock_vm = MagicMock()
    mock_vm.name = 'test-vm'
    mock_vm.get_vm_ip.return_value = '192.168.1.102'

    mock_ssh_client_instance = MagicMock()
    mock_ssh_client_cls.return_value = mock_ssh_client_instance
    mock_ssh_client_instance.execute_command.return_value = True

    manager = GitHubRunnerManager(mock_runner_config)
    assert manager.register_runner(mock_vm) is True
    mock_ssh_client_instance.execute_command.assert_called_once_with(
        'cd actions-runner && ./config.sh --url https://github.com/test_org/test_repo --token mock_token --name test-vm --ephemeral --unattended --labels macos,tart',
        RUNNER_CONFIG_TIMEOUT,
    )
    mock_thread.assert_called_once()
    mock_sleep.assert_called_once_with(RUNNER_START_DELAY)


@pytest.fixture
def mock_image_config():
    return MagicMock(
        base_image='test_base_image',
        runner_image='test_runner_image',
        convert_command='mock_convert_cmd',
        update_hour=DEFAULT_UPDATE_HOUR,
    )


@pytest.fixture
def mock_update_file(tmp_path):
    update_file = tmp_path / '.tarty_last_update.json'
    with patch('pathlib.Path.home', return_value=tmp_path):
        yield update_file


def test_image_manager_load_last_update_success(mock_update_file):
    test_time = datetime(2023, 1, 1, 10, 0, 0)
    mock_update_file.write_text(
        json.dumps({'last_update': test_time.isoformat()})
    )
    manager = ImageManager(MagicMock())
    assert manager.last_update == test_time


def test_image_manager_load_last_update_no_file(mock_update_file):
    manager = ImageManager(MagicMock())
    assert manager.last_update is None


def test_image_manager_save_last_update_success(mock_update_file):
    manager = ImageManager(MagicMock())
    test_time = datetime(2023, 1, 2, 12, 30, 0)
    manager.last_update = test_time
    manager._save_last_update()
    with open(mock_update_file, 'r') as f:
        data = json.load(f)
    assert datetime.fromisoformat(data['last_update']) == test_time


@patch('tarty.run_tart_command')
def test_create_runner_image_success(
    mock_run_tart_command, mock_image_config
):
    manager = ImageManager(mock_image_config)
    assert manager.create_runner_image('new-vm') is True
    mock_run_tart_command.assert_called_once_with(
        ['clone', 'test_runner_image', 'new-vm'],
        check=True,
        capture_output=True,
    )


@patch(
    'tarty.run_tart_command',
    side_effect=subprocess.CalledProcessError(1, 'cmd'),
)
def test_create_runner_image_failure(
    mock_run_tart_command, mock_image_config
):
    manager = ImageManager(mock_image_config)
    assert manager.create_runner_image('new-vm') is False


@patch('tarty.ImageManager._load_last_update', return_value=None)
def test_should_update_never_updated(
    mock_load_last_update, mock_image_config
):
    manager = ImageManager(mock_image_config)
    assert manager.should_update() is True


@patch('tarty.ImageManager._load_last_update')
@patch('tarty.datetime')
def test_should_update_force_update(
    mock_datetime, mock_load_last_update, mock_image_config
):
    mock_datetime.now.return_value = datetime(2023, 1, 2, 10, 0, 0)
    mock_load_last_update.return_value = datetime(
        2023, 1, 1, 10, 0, 0
    )  # 24 hours ago
    manager = ImageManager(mock_image_config)
    assert manager.should_update() is True


@patch('tarty.ImageManager._load_last_update')
@patch('tarty.datetime')
def test_should_update_nightly_update(
    mock_datetime, mock_load_last_update, mock_image_config
):
    mock_datetime.now.return_value = datetime(2023, 1, 2, 2, 0, 0)
    mock_load_last_update.return_value = datetime(2023, 1, 1, 15, 0, 0)
    manager = ImageManager(mock_image_config)
    assert manager.should_update() is True


@patch('tarty.ImageManager._load_last_update')
@patch('tarty.datetime')
def test_should_update_no_update_needed(
    mock_datetime, mock_load_last_update, mock_image_config
):
    mock_datetime.now.return_value = datetime(2023, 1, 2, 10, 0, 0)
    mock_load_last_update.return_value = datetime(2023, 1, 2, 9, 0, 0)
    manager = ImageManager(mock_image_config)
    assert manager.should_update() is False


@patch('tarty.ImageManager._load_last_update')
@patch('tarty.datetime')
def test_should_update_no_update_needed_same_day_same_hour(
    mock_datetime, mock_load_last_update, mock_image_config
):
    mock_datetime.now.return_value = datetime(
        2023, 1, 2, DEFAULT_UPDATE_HOUR, 30, 0
    )
    mock_load_last_update.return_value = datetime(
        2023, 1, 2, DEFAULT_UPDATE_HOUR, 0, 0
    )
    manager = ImageManager(mock_image_config)
    assert manager.should_update() is False


@patch('tarty.ImageManager._delete_old_runner_image')
@patch('tarty.ImageManager._run_conversion', return_value=True)
@patch('tarty.ImageManager._create_temp_vm')
@patch('tarty.run_tart_command')
@patch('tarty.ImageManager._save_last_update')
@patch('tarty.time.time', return_value=1234567890.0)
def test_update_runner_image_success(
    mock_time,
    mock_save_last_update,
    mock_run_tart_command,
    mock_create_temp_vm,
    mock_run_conversion,
    mock_delete_old_runner_image,
    mock_image_config,
):
    mock_image_config.convert_command = 'some_command'

    temp_vm_name_expected = f'tarty-convert-{int(mock_time.return_value)}'

    mock_temp_vm = MagicMock()
    mock_temp_vm.name = temp_vm_name_expected
    mock_temp_vm.stop.return_value = True
    mock_create_temp_vm.return_value = mock_temp_vm

    manager = ImageManager(mock_image_config)
    manager.last_update = datetime(2023, 1, 1)

    assert manager.update_runner_image() is True

    mock_create_temp_vm.assert_called_once_with(temp_vm_name_expected)
    mock_run_conversion.assert_called_once_with(mock_temp_vm)
    mock_temp_vm.stop.assert_called_once()
    mock_delete_old_runner_image.assert_called_once()
    mock_run_tart_command.assert_called_once_with(
        ['rename', temp_vm_name_expected, mock_image_config.runner_image],
        check=True,
        capture_output=True,
    )


@patch('tarty.ImageManager._delete_old_runner_image')
@patch('tarty.ImageManager._run_conversion', return_value=False)
@patch('tarty.ImageManager._create_temp_vm')
@patch('tarty.run_tart_command')
@patch('tarty.ImageManager._cleanup_temp_vm')
@patch('tarty.time.time', return_value=1234567890.0)
def test_update_runner_image_conversion_fails(
    mock_time,
    mock_cleanup_temp_vm,
    mock_run_tart_command,
    mock_create_temp_vm,
    mock_run_conversion,
    mock_delete_old_runner_image,
    mock_image_config,
):
    mock_image_config.convert_command = 'some_command'

    temp_vm_name_expected = f'tarty-convert-{int(mock_time.return_value)}'
    mock_temp_vm = MagicMock()
    mock_temp_vm.name = temp_vm_name_expected
    mock_create_temp_vm.return_value = mock_temp_vm

    manager = ImageManager(mock_image_config)
    assert manager.update_runner_image() is False

    mock_create_temp_vm.assert_called_once_with(temp_vm_name_expected)
    mock_run_conversion.assert_called_once_with(mock_temp_vm)
    mock_cleanup_temp_vm.assert_called_once_with(temp_vm_name_expected)
    mock_run_tart_command.assert_not_called()


@patch('tarty.GitHubRunnerManager')
@patch('tarty.ImageManager')
def test_orchestrator_init_success(
    mock_image_manager_cls, mock_runner_manager_cls
):
    mock_config = MagicMock()

    orchestrator = TartyOrchestrator(mock_config)
    assert orchestrator.config == mock_config
    assert isinstance(orchestrator.runner_manager, MagicMock)
    assert isinstance(orchestrator.image_manager, MagicMock)
    assert orchestrator.running is False
    assert orchestrator.vms == {}


@patch('tarty.GitHubRunnerManager', side_effect=Exception('Test error'))
@patch('tarty.ImageManager')
@patch('sys.exit')
def test_orchestrator_init_config_error(
    mock_sys_exit, mock_image_manager_cls, mock_runner_manager_cls
):
    mock_config = MagicMock()
    TartyOrchestrator(mock_config)
    mock_sys_exit.assert_called_once_with(1)


@patch('tarty.cleanup_previous_tarty_vms')
@patch('tarty.TartyOrchestrator._orchestration_cycle')
@patch('time.sleep', return_value=None)
def test_orchestrator_start_stop_loop(mock_sleep, mock_cycle, mock_cleanup):
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        mock_cycle.side_effect = [None, None, KeyboardInterrupt]

        orchestrator.start()

        mock_cleanup.assert_called_once()
        assert mock_cycle.call_count == 3
        assert orchestrator.running is False
        mock_sleep.assert_has_calls(
            [
                call(ORCHESTRATION_CYCLE_INTERVAL),
                call(ORCHESTRATION_CYCLE_INTERVAL),
            ]
        )


@patch('tarty.TartyOrchestrator._cleanup_completed_vms')
@patch('tarty.TartyOrchestrator._start_new_vms_if_needed')
@patch('tarty.TartyOrchestrator._check_nightly_updates')
def test_orchestration_cycle_calls_sub_methods(
    mock_check_updates, mock_start_vms, mock_cleanup_vms
):
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)
        orchestrator._orchestration_cycle()
        mock_check_updates.assert_called_once()
        mock_cleanup_vms.assert_called_once()
        mock_start_vms.assert_called_once()


@patch('tarty.TartyOrchestrator._cleanup_vm')
def test_cleanup_completed_vms(mock_cleanup_vm):
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        vm1 = MagicMock()
        vm1.name = 'vm1'
        vm1.running = True
        vm2 = MagicMock()
        vm2.name = 'vm2'
        vm2.running = False
        vm3 = MagicMock()
        vm3.name = 'vm3'
        vm3.running = False

        orchestrator.vms = {'vm1': vm1, 'vm2': vm2, 'vm3': vm3}

        orchestrator._cleanup_completed_vms()

        mock_cleanup_vm.assert_any_call(vm2)
        mock_cleanup_vm.assert_any_call(vm3)
        assert mock_cleanup_vm.call_count == 2


@patch(
    'tarty.TartyOrchestrator._create_and_start_vm',
    side_effect=[True, False],
)
@patch('tarty.time.time', return_value=1234567890.0)
def test_start_new_vms_if_needed(mock_time, mock_create_and_start_vm):
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        mock_config.max_vms = 2

        orchestrator = TartyOrchestrator(mock_config)
        orchestrator.vms = {}

        orchestrator._start_new_vms_if_needed()

        mock_create_and_start_vm.assert_called_once()
        assert mock_create_and_start_vm.call_args[0][0].startswith(
            'tarty-runner-'
        )


@patch('tarty.time.time', return_value=1234567890.0)
def test_create_and_start_vm_success(mock_time):
    with (
        patch('tarty.GitHubRunnerManager') as mock_runner_manager_cls,
        patch('tarty.ImageManager') as mock_image_manager_cls,
    ):
        mock_runner_manager_instance = mock_runner_manager_cls.return_value
        mock_image_manager_instance = mock_image_manager_cls.return_value

        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        mock_vm_instance = MagicMock()
        mock_vm_instance.name = 'test-vm-instance'
        mock_vm_instance.start.return_value = True
        with patch(
            'tarty.TartVM', return_value=mock_vm_instance
        ) as mock_tart_vm_cls:
            mock_image_manager_instance.create_runner_image.return_value = (
                True
            )
            mock_runner_manager_instance.register_runner.return_value = True

            vm_name = f'tarty-runner-{int(mock_time.return_value)}'
            assert orchestrator._create_and_start_vm(vm_name) is True

            mock_image_manager_instance.create_runner_image.assert_called_once_with(
                vm_name
            )
            mock_tart_vm_cls.assert_called_once_with(vm_name, vm_name)
            mock_vm_instance.start.assert_called_once()
            mock_runner_manager_instance.register_runner.assert_called_once_with(
                mock_vm_instance
            )
            assert orchestrator.vms[vm_name] == mock_vm_instance


@patch('tarty.time.time', return_value=1234567890.0)
def test_create_and_start_vm_image_creation_fail(mock_time):
    with (
        patch('tarty.GitHubRunnerManager') as mock_runner_manager_cls,
        patch('tarty.ImageManager') as mock_image_manager_cls,
    ):
        mock_runner_manager_instance = mock_runner_manager_cls.return_value
        mock_image_manager_instance = mock_image_manager_cls.return_value

        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        mock_image_manager_instance.create_runner_image.return_value = False

        vm_name = f'tarty-runner-{int(mock_time.return_value)}'
        assert orchestrator._create_and_start_vm(vm_name) is False

        mock_image_manager_instance.create_runner_image.assert_called_once_with(
            vm_name
        )
        mock_runner_manager_instance.register_runner.assert_not_called()
        assert vm_name not in orchestrator.vms


@patch('tarty.time.time', return_value=1234567890.0)
def test_create_and_start_vm_runner_registration_fail(mock_time):
    with (
        patch('tarty.GitHubRunnerManager') as mock_runner_manager_cls,
        patch('tarty.ImageManager') as mock_image_manager_cls,
    ):
        mock_runner_manager_instance = mock_runner_manager_cls.return_value
        mock_image_manager_instance = mock_image_manager_cls.return_value

        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        mock_vm_instance = MagicMock()
        mock_vm_instance.name = 'test-vm-instance'
        mock_vm_instance.start.return_value = True
        with patch(
            'tarty.TartVM', return_value=mock_vm_instance
        ) as mock_tart_vm_cls:
            mock_image_manager_instance.create_runner_image.return_value = (
                True
            )
            mock_runner_manager_instance.register_runner.return_value = False

            vm_name = f'tarty-runner-{int(mock_time.return_value)}'
            assert orchestrator._create_and_start_vm(vm_name) is False

            mock_image_manager_instance.create_runner_image.assert_called_once_with(
                vm_name
            )
            mock_tart_vm_cls.assert_called_once_with(vm_name, vm_name)
            mock_vm_instance.start.assert_called_once()
            mock_runner_manager_instance.register_runner.assert_called_once_with(
                mock_vm_instance
            )
            mock_vm_instance.destroy.assert_called_once()
            assert vm_name not in orchestrator.vms


def test_cleanup_vm():
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        mock_vm = MagicMock()
        mock_vm.name = 'test-vm'
        mock_vm.destroy.return_value = True
        orchestrator.vms = {'test-vm': mock_vm}

        orchestrator._cleanup_vm(mock_vm)

        mock_vm.destroy.assert_called_once()
        assert 'test-vm' not in orchestrator.vms


@patch('tarty.TartyOrchestrator._make_room_for_update')
@patch('tarty.datetime')
def test_check_nightly_updates_performs_update(
    mock_datetime,
    mock_make_room,
):
    mock_now = datetime(2023, 1, 2, 10, 0, 0)
    mock_datetime.now.return_value = mock_now

    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager') as mock_image_manager_cls,
    ):
        mock_config = MagicMock()
        mock_config.update_hour = 10

        mock_image_manager_instance = mock_image_manager_cls.return_value
        mock_image_manager_instance.should_update.return_value = True
        mock_image_manager_instance.update_runner_image.return_value = True
        mock_image_manager_instance._save_last_update.return_value = None

        orchestrator = TartyOrchestrator(mock_config)
        orchestrator.last_update_check = None

        orchestrator._check_nightly_updates()

        mock_image_manager_instance.should_update.assert_called_once_with(
            mock_config.update_hour
        )
        mock_make_room.assert_called_once()
        mock_image_manager_instance.update_runner_image.assert_called_once()
        assert orchestrator.image_manager.last_update == mock_now
        mock_image_manager_instance._save_last_update.assert_called_once()
        assert orchestrator.last_update_check == mock_now


@patch('tarty.TartyOrchestrator._make_room_for_update')
@patch('tarty.datetime')
def test_check_nightly_updates_skips_update(
    mock_datetime,
    mock_make_room,
):
    mock_now = datetime(2023, 1, 2, 10, 0, 0)
    mock_datetime.now.return_value = mock_now

    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager') as mock_image_manager_cls,
    ):
        mock_config = MagicMock()
        mock_config.update_hour = 10

        mock_image_manager_instance = mock_image_manager_cls.return_value
        mock_image_manager_instance.should_update.return_value = False
        mock_image_manager_instance.update_runner_image.return_value = True

        orchestrator = TartyOrchestrator(mock_config)
        orchestrator.last_update_check = None

        orchestrator._check_nightly_updates()

        mock_image_manager_instance.should_update.assert_called_once_with(
            mock_config.update_hour
        )
        mock_make_room.assert_not_called()
        mock_image_manager_instance.update_runner_image.assert_not_called()
        assert orchestrator.last_update_check == mock_now


@patch('tarty.TartyOrchestrator._cleanup_vm')
def test_make_room_for_update_kills_one_vm(mock_cleanup_vm):
    with (
        patch('tarty.GitHubRunnerManager'),
        patch('tarty.ImageManager'),
    ):
        mock_config = MagicMock()
        orchestrator = TartyOrchestrator(mock_config)

        vm1 = MagicMock(name='vm1')
        vm2 = MagicMock(name='vm2')
        orchestrator.vms = {'vm1': vm1, 'vm2': vm2}

        orchestrator._make_room_for_update()

        mock_cleanup_vm.assert_called_once()
        assert mock_cleanup_vm.call_args[0][0] in [vm1, vm2]
