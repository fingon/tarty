#!/usr/bin/env -S uv run --python python3.13 --with pydantic python3
# -*- coding: utf-8 -*-
# -*- Python -*-
"""
Tarty - Tart VM Manager for GitHub Runners

This tool orchestrates up to 2 virtualized macOS VMs using tart,
managing their lifecycle as ephemeral GitHub runners.
"""

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator

# Initialize module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_USERNAME = 'admin'
VM_START_TIMEOUT = 60
ORCHESTRATION_CYCLE_INTERVAL = 30
UPDATE_CHECK_INTERVAL = 3600
MAX_VMS = 2
DEFAULT_UPDATE_HOUR = 2
RUNNER_CONFIG_TIMEOUT = 60
RUNNER_START_TIMEOUT = 30
RUNNER_START_DELAY = 5
IMAGE_CONVERSION_TIMEOUT = 600


# Enums
class VMState(Enum):
    """VM state enumeration."""

    STOPPED = 'stopped'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'


# Exceptions
class TartyError(Exception):
    """Base exception for Tarty operations."""

    pass


class VMStartError(TartyError):
    """VM failed to start."""

    pass


class RunnerRegistrationError(TartyError):
    """Runner registration failed."""

    pass


class ImageUpdateError(TartyError):
    """Image update failed."""

    pass


class ConfigError(TartyError):
    """Configuration error."""

    pass


class RunnerConfig(BaseModel):
    """Configuration schema for GitHub runner settings."""

    github_pat: str
    organization: str
    base_image: str
    runner_image: str
    ssh_username: str = DEFAULT_SSH_USERNAME
    update_hour: int = DEFAULT_UPDATE_HOUR
    max_vms: int = MAX_VMS
    convert_command: str | None = None
    repository: str | None = None
    labels: list[str] | None = None
    ephemeral: bool = True
    vm_prefix: str = 'tarty'

    @field_validator(
        'github_pat',
        'organization',
        'base_image',
        'runner_image',
    )
    @classmethod
    def check_not_empty(cls, v):
        if not v:
            raise ValueError('Field cannot be empty')
        return v

    @field_validator('update_hour')
    @classmethod
    def check_update_hour_range(cls, v):
        if not (0 <= v <= 23):
            raise ValueError('update_hour must be between 0 and 23')
        return v

    @field_validator('max_vms')
    @classmethod
    def check_max_vms_range(cls, v):
        if not (1 <= v <= 2):
            raise ValueError('max_vms must be 1 or 2')
        return v


# Utility functions
def run_tart_command(args: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a tart command with given arguments."""
    return subprocess.run(['tart'] + args, **kwargs)


def run_ssh_command(
    host: str, username: str, command: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run a command via SSH."""
    ssh_cmd = [
        'ssh',
        '-o',
        'StrictHostKeyChecking=no',
        '-o',
        'UserKnownHostsFile=/dev/null',
        f'{username}@{host}',
        command,
    ]
    return subprocess.run(
        ssh_cmd, capture_output=True, text=True, timeout=timeout
    )


def cleanup_previous_vms(vm_prefix: str):
    """Delete all previous VMs with the given prefix."""
    try:
        result = run_tart_command(
            ['list', '-q'], capture_output=True, text=True, check=True
        )
        vm_names = result.stdout.strip().split('\n')

        for vm_name in vm_names:
            vm_name = vm_name.strip()
            if vm_name.startswith(f'{vm_prefix}-'):
                logger.info('Cleaning up previous VM: %s', vm_name)
                try:
                    run_tart_command(
                        ['delete', vm_name], capture_output=True, check=True
                    )
                    logger.info('Successfully deleted VM: %s', vm_name)
                except subprocess.CalledProcessError as e:
                    logger.warning('Failed to delete VM %s: %s', vm_name, e)

    except subprocess.CalledProcessError as e:
        logger.error('Failed to list VMs for cleanup: %s', e)
    except Exception as e:
        logger.error('Error during VM cleanup: %s', e)


class SSHClient:
    """SSH client for VM operations."""

    def __init__(self, host: str, username: str):
        self.host = host
        self.username = username

    def execute_command(self, command: str, timeout: int = 60) -> bool:
        """Execute a command via SSH and return success status."""
        try:
            result = run_ssh_command(
                self.host, self.username, command, timeout
            )
            if result.returncode == 0:
                logger.info('SSH command succeeded on %s', self.host)
                return True
            else:
                logger.error(
                    'SSH command failed on %s: %s', self.host, result.stderr
                )
                return False
        except Exception as e:
            logger.error('SSH command execution failed: %s', e)
            return False

    def is_available(self, *, timeout: int, retry: int = 1) -> bool:
        """Check if SSH is available on the host."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.execute_command('echo running', timeout=1):
                return True
            time.sleep(retry)
        return False


class TartyConfig:
    """Configuration management for tarty."""

    def __init__(self, config_path: str, cli_args=None):
        self.config_path = config_path
        self.cli_args = cli_args
        self.runner_config = self._load_config()

    def _load_config(self) -> RunnerConfig:
        """Load configuration from file and apply CLI overrides."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(
                'Configuration file not found: ' + self.config_path
            )
        except json.JSONDecodeError as e:
            raise ConfigError('Invalid JSON in config file: ' + str(e))

        # Direct rewrite from CLI args
        if self.cli_args:
            for key, value in vars(self.cli_args).items():
                if value is not None and key.replace('_', '-') in [
                    'github-pat',
                    'organization',
                    'repository',
                    'base-image',
                    'runner-image',
                    'ssh-username',
                    'convert-command',
                    'update-hour',
                    'max-vms',
                    'labels',
                    'vm-prefix',
                ]:
                    config_data[key.replace('-', '_')] = value

            # Handle ephemeral flag specially
            if self.cli_args.ephemeral:
                config_data['ephemeral'] = True
            elif self.cli_args.no_ephemeral:
                config_data['ephemeral'] = False

        try:
            return RunnerConfig(**config_data)
        except Exception as e:
            raise ConfigError('Invalid configuration: ' + str(e))

    @property
    def github_pat(self) -> str:
        return self.runner_config.github_pat

    @property
    def organization(self) -> str:
        return self.runner_config.organization

    @property
    def repository(self) -> str:
        return self.runner_config.repository

    @property
    def base_image(self) -> str:
        return self.runner_config.base_image

    @property
    def runner_image(self) -> str:
        return self.runner_config.runner_image

    @property
    def ssh_username(self) -> str:
        return self.runner_config.ssh_username

    @property
    def convert_command(self) -> str:
        return self.runner_config.convert_command

    @property
    def update_hour(self) -> int:
        return self.runner_config.update_hour

    @property
    def max_vms(self) -> int:
        return self.runner_config.max_vms

    @property
    def labels(self) -> list[str] | None:
        return self.runner_config.labels

    @property
    def ephemeral(self) -> bool:
        return self.runner_config.ephemeral

    @property
    def vm_prefix(self) -> str:
        return self.runner_config.vm_prefix


class TartVM:
    """Represents a single tart VM instance."""

    def __init__(self, name: str, image: str):
        self.name = name
        self.image = image
        self.state = VMState.STOPPED
        self.runner_token = None
        self.process = None
        self._ip_address = None
        self.runner_process = None

    @property
    def running_vm(self) -> bool:
        """Check if VM is running."""
        if self.state != VMState.RUNNING:
            return False

        try:
            result = run_tart_command(
                ['get', self.name, '--format', 'json'],
                capture_output=True,
                text=True,
                check=True,
            )
            vm_info = json.loads(result.stdout)
            return vm_info.get('Running', False)
        except subprocess.CalledProcessError:
            return False

    @property
    def running(self) -> bool:
        """Check if VM is running and runner is active."""
        if not self.running_vm:
            return False

        # Check if runner process is still alive
        if self.runner_process and not self.runner_process.is_alive():
            logger.info('Runner process completed for VM: %s', self.name)
            return False

        return True

    def start(self) -> bool:
        """Start the VM and wait for SSH to be available."""
        try:
            logger.info('VM start: %s', self.name)
            self.state = VMState.STARTING

            self._start_vm_process()

            # Wait for SSH to be available
            vm_ip = self.get_vm_ip()
            if not vm_ip:
                raise VMStartError('Failed to get IP for VM ' + self.name)

            ssh_client = SSHClient(
                vm_ip, 'admin'
            )  # Will be configurable later
            if not ssh_client.is_available(timeout=VM_START_TIMEOUT):
                raise VMStartError('SSH not available for VM ' + self.name)

            self.state = VMState.RUNNING
            self._ip_address = vm_ip
            logger.info(
                'VM start succeeded: %s - SSH available on %s',
                self.name,
                vm_ip,
            )
            return True

        except (VMStartError, subprocess.CalledProcessError) as e:
            logger.error('VM start failed: %s - %s', self.name, str(e))
            self.stop()
            return False

    def _start_vm_process(self):
        """Start the VM process in a separate thread."""

        def run_vm():
            try:
                self.process = run_tart_command(
                    ['run', '--no-graphics', self.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.error('VM process failed: %s', e)

        vm_thread = threading.Thread(target=run_vm, daemon=True)
        vm_thread.start()

    def get_vm_ip(self) -> str | None:
        """Get the IP address of the VM."""
        if self._ip_address:
            return self._ip_address

        logger.info('Starting to get ip of %s', self.name)
        start_time = time.time()
        timeout = 30
        last_e = None
        while time.time() - start_time < timeout:
            try:
                result = run_tart_command(
                    ['ip', self.name],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self._ip_address = result.stdout.strip()
                logger.info('ip of %s is %s', self.name, self._ip_address)
                return self._ip_address
            except subprocess.CalledProcessError as e:
                last_e = e
                time.sleep(1)
        stderr_output = last_e.stderr if last_e.stderr else 'No stderr output'
        logger.error(
            'Failed to get VM IP after %ss: %s - stderr: %s',
            timeout,
            last_e,
            stderr_output,
        )
        return None

    def stop(self) -> bool:
        """Stop the VM."""
        try:
            if self.state == VMState.STOPPED:
                return True

            logger.info('VM stop attempting: %s', self.name)
            self.state = VMState.STOPPING

            run_tart_command(
                ['stop', self.name], check=True, capture_output=True
            )

            self.state = VMState.STOPPED
            self.process = None
            self._ip_address = None
            logger.info('VM stop succeeded: %s', self.name)
            return True

        except subprocess.CalledProcessError as e:
            logger.error('VM stop failed: %s - %s', self.name, str(e))
            # Even if tart stop fails, we consider the VM stopped from our perspective
            # as we can't interact with it further.
            self.state = VMState.STOPPED
            self.process = None
            self._ip_address = None
            return False

    def destroy(self) -> bool:
        """Destroy the VM."""
        try:
            if self.running_vm:
                self.stop()

            logger.info('VM destroy attempting: %s', self.name)

            run_tart_command(
                ['delete', self.name], check=True, capture_output=True
            )

            logger.info('VM destroy succeeded: %s', self.name)
            return True

        except subprocess.CalledProcessError as e:
            logger.error('VM destroy failed: %s - %s', self.name, str(e))
            return False


class GitHubRunnerManager:
    """Manages GitHub runner registration and lifecycle."""

    def __init__(self, config: TartyConfig):
        self.config = config

    def get_registration_token(self) -> str | None:
        """Get a registration token from GitHub API."""
        try:
            if self.config.repository:
                url = (
                    'https://api.github.com/repos/'
                    + self.config.organization
                    + '/'
                    + self.config.repository
                    + '/actions/runners/registration-token'
                )
            else:
                url = (
                    'https://api.github.com/orgs/'
                    + self.config.organization
                    + '/actions/runners/registration-token'
                )
            req = urllib.request.Request(url, method='POST')
            req.add_header('Authorization', 'token ' + self.config.github_pat)
            req.add_header('Accept', 'application/vnd.github.v3+json')

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                token = data.get('token')
                if token:
                    logger.info(
                        'Successfully obtained GitHub registration token'
                    )
                return token

        except Exception as e:
            logger.error('Failed to get registration token: %s', e)
            return None

    def register_runner(self, vm: TartVM) -> bool:
        """Register and start a runner with GitHub."""
        try:
            token = self.get_registration_token()
            if not token:
                raise RunnerRegistrationError(
                    'Failed to get registration token'
                )

            vm_ip = vm.get_vm_ip()
            if not vm_ip:
                raise RunnerRegistrationError(
                    'Failed to get IP for VM ' + vm.name
                )

            ssh_client = SSHClient(vm_ip, self.config.ssh_username)

            if not self._configure_runner(ssh_client, token, vm.name):
                raise RunnerRegistrationError('Failed to configure runner')

            if not self._start_runner_process(ssh_client, vm):
                raise RunnerRegistrationError('Failed to start runner')

            vm.runner_token = token
            logger.info('Runner registered and started for VM: %s', vm.name)
            return True

        except RunnerRegistrationError as e:
            logger.error(
                'Failed to register runner for VM %s: %s', vm.name, e
            )
            return False

    def _configure_runner(
        self, ssh_client: SSHClient, token: str, runner_name: str
    ) -> bool:
        """Configure the GitHub runner via SSH."""
        url = 'https://github.com/' + self.config.organization
        if self.config.repository:
            url += '/' + self.config.repository

        command = f'cd actions-runner && ./config.sh --url {url} --token {token} --name {runner_name}'

        if self.config.ephemeral:
            command += ' --ephemeral'
        else:
            command += ' --replace'

        command += ' --unattended'

        if self.config.labels and len(self.config.labels) > 0:
            labels_str = ','.join(self.config.labels)
            command += f' --labels {labels_str}'

        logger.info('Configuring runner on %s', ssh_client.host)
        return ssh_client.execute_command(command, RUNNER_CONFIG_TIMEOUT)

    def _start_runner_process(
        self, ssh_client: SSHClient, vm: TartVM
    ) -> bool:
        """Start the GitHub runner process via SSH in a separate thread."""

        def run_runner():
            try:
                command = 'cd actions-runner && ./run.sh'
                logger.info('Starting runner process on %s', ssh_client.host)
                result = run_ssh_command(
                    ssh_client.host,
                    ssh_client.username,
                    command,
                    timeout=None,  # No timeout for long-running process
                )
                logger.info(
                    'Runner process completed on %s with exit code %d',
                    ssh_client.host,
                    result.returncode,
                )
            except Exception as e:
                logger.error(
                    'Runner process failed on %s: %s', ssh_client.host, e
                )

        runner_thread = threading.Thread(target=run_runner, daemon=True)
        runner_thread.start()
        vm.runner_process = runner_thread

        # Give the runner a moment to start
        time.sleep(RUNNER_START_DELAY)
        return True


class ImageManager:
    """Manages tart image updates and creation."""

    def __init__(self, config: TartyConfig):
        self.config = config
        self.last_update = self._load_last_update()

    def _get_update_file_path(self) -> Path:
        """Get the path to the last update timestamp file."""
        return Path.home() / '.tarty_last_update.json'

    def _load_last_update(self) -> datetime | None:
        """Load last update time from file."""
        try:
            update_file = self._get_update_file_path()
            with update_file.open('r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_update'])
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def _save_last_update(self):
        """Save last update time to file."""
        try:
            update_file = self._get_update_file_path()
            with update_file.open('w') as f:
                json.dump({'last_update': self.last_update.isoformat()}, f)
        except Exception as e:
            logger.error('Failed to save last update time: %s', e)

    def create_runner_image(self, vm_name: str) -> bool:
        """Create a runner image from the runner base image."""
        try:
            logger.info('Creating runner image: %s', vm_name)
            run_tart_command(
                ['clone', self.config.runner_image, vm_name],
                check=True,
                capture_output=True,
            )
            logger.info('Runner image %s created successfully', vm_name)
            return True
        except subprocess.CalledProcessError as e:
            logger.error('Failed to create runner image %s: %s', vm_name, e)
            return False

    def update_runner_image(self) -> bool:
        """Update the runner image by converting base image."""
        if not self.config.convert_command:
            logger.info(
                'No convert command specified, skipping runner image update'
            )
            return True

        temp_vm_name = f'{self.config.vm_prefix}-convert-' + str(
            int(time.time())
        )

        try:
            logger.info('Starting runner image update process')

            temp_vm = self._create_temp_vm(temp_vm_name)
            if not temp_vm:
                return False

            if not self._run_conversion(temp_vm):
                self._cleanup_temp_vm(temp_vm_name)
                return False

            return self._finalize_image_update(temp_vm, temp_vm_name)

        except Exception as e:
            logger.error('Failed to update runner image: %s', e)
            self._cleanup_temp_vm(temp_vm_name)
            return False

    def _create_temp_vm(self, temp_vm_name: str) -> TartVM | None:
        """Create and start temporary VM for conversion."""
        try:
            logger.info(
                'Cloning base image to temporary VM: %s', temp_vm_name
            )
            run_tart_command(
                ['clone', self.config.base_image, temp_vm_name],
                check=True,
                capture_output=True,
            )

            temp_vm = TartVM(temp_vm_name, temp_vm_name)
            if not temp_vm.start():
                raise ImageUpdateError(
                    'Failed to start temporary VM for conversion'
                )

            return temp_vm

        except (subprocess.CalledProcessError, ImageUpdateError) as e:
            logger.error('Failed to create temporary VM: %s', e)
            return None

    def _run_conversion(self, temp_vm: TartVM) -> bool:
        """Run conversion command on temporary VM."""
        vm_ip = temp_vm.get_vm_ip()
        if not vm_ip:
            logger.error('Failed to get IP for temporary VM')
            return False

        logger.info('Running conversion command on %s', vm_ip)
        ssh_client = SSHClient(vm_ip, self.config.ssh_username)

        if not ssh_client.execute_command(
            self.config.convert_command, IMAGE_CONVERSION_TIMEOUT
        ):
            logger.error('Conversion command failed')
            return False

        logger.info('Conversion command completed successfully')
        return True

    def _finalize_image_update(
        self, temp_vm: TartVM, temp_vm_name: str
    ) -> bool:
        """Finalize the image update by stopping VM and renaming."""
        try:
            temp_vm.stop()

            self._delete_old_runner_image()

            run_tart_command(
                ['rename', temp_vm_name, self.config.runner_image],
                check=True,
                capture_output=True,
            )

            logger.info('Runner image updated successfully')
            return True

        except subprocess.CalledProcessError as e:
            logger.error('Failed to finalize image update: %s', e)
            return False

    def _delete_old_runner_image(self):
        """Delete old runner image if it exists."""
        try:
            run_tart_command(
                ['delete', self.config.runner_image],
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            pass  # Image might not exist

    def _cleanup_temp_vm(self, temp_vm_name: str):
        """Clean up temporary VM."""
        try:
            run_tart_command(['delete', temp_vm_name], capture_output=True)
        except subprocess.CalledProcessError:
            pass

    def should_update(self, update_hour: int = DEFAULT_UPDATE_HOUR) -> bool:
        """Check if runner image should be updated (nightly at specified hour)."""
        now = datetime.now()

        # If never updated, update now
        if not self.last_update:
            return True

        # If more than 22 hours since last update, force update regardless of hour
        if now - self.last_update > timedelta(hours=22):
            return True

        # If it's the right hour and we haven't updated today
        if now.hour == update_hour and now.date() > self.last_update.date():
            return True

        return False


class TartyOrchestrator:
    """Main orchestrator for managing VMs and runners."""

    def __init__(self, config: TartyConfig):
        try:
            self.config = config
            self.runner_manager = GitHubRunnerManager(self.config)
            self.image_manager = ImageManager(self.config)
            self.vms: dict[str, TartVM] = {}
            self.running = False
            self.last_update_check = None
        except Exception as e:
            logger.error('Configuration error: %s', e)
            sys.exit(1)

    def start(self):
        """Start the orchestrator."""
        logger.info('Starting Tarty orchestrator')

        cleanup_previous_vms(self.config.vm_prefix)

        self.running = True

        try:
            while self.running:
                self._orchestration_cycle()
                time.sleep(ORCHESTRATION_CYCLE_INTERVAL)
        except KeyboardInterrupt:
            logger.info('Received interrupt signal')
        finally:
            self.stop()

    def stop(self):
        """Stop the orchestrator and clean up."""
        logger.info('Stopping Tarty orchestrator')
        self.running = False

        for vm in list(self.vms.values()):
            self._cleanup_vm(vm)

    def _orchestration_cycle(self):
        """Single cycle of the orchestration loop."""
        try:
            self._check_nightly_updates()

            self._cleanup_completed_vms()

            self._start_new_vms_if_needed()

        except Exception as e:
            logger.error('Error in orchestration cycle: %s', e)

    def _cleanup_completed_vms(self):
        """Remove completed VMs from tracking."""
        completed_vms = [vm for vm in self.vms.values() if not vm.running]
        for vm in completed_vms:
            self._cleanup_vm(vm)

    def _start_new_vms_if_needed(self):
        """Start new VMs if we have capacity."""
        if self.config.ephemeral:
            while len(self.vms) < self.config.max_vms:
                vm_name = f'{self.config.vm_prefix}-runner-' + str(
                    int(time.time())
                )
                if self._create_and_start_vm(vm_name):
                    break  # Only create one VM per cycle
        else:
            # For non-ephemeral mode, create VMs with fixed names
            for i in range(self.config.max_vms):
                vm_name = f'{self.config.vm_prefix}-runner-{i + 1}'
                if vm_name not in self.vms:
                    self._create_and_start_vm(vm_name)
                    break  # Only create one VM per cycle

    def _create_and_start_vm(self, vm_name: str) -> bool:
        """Create and start a new VM."""
        try:
            if not self.image_manager.create_runner_image(vm_name):
                return False

            vm = TartVM(vm_name, vm_name)

            if not vm.start():
                vm.destroy()
                return False

            if not self.runner_manager.register_runner(vm):
                vm.destroy()
                return False

            self.vms[vm_name] = vm
            logger.info('Successfully created and started VM: %s', vm_name)
            return True

        except Exception as e:
            logger.error('Failed to create and start VM %s: %s', vm_name, e)
            return False

    def _cleanup_vm(self, vm: TartVM):
        """Clean up a VM."""
        logger.info('VM cleanup attempting: %s', vm.name)

        success = vm.destroy()

        if vm.name in self.vms:
            del self.vms[vm.name]

        if success:
            logger.info('VM cleanup succeeded: %s', vm.name)
        else:
            logger.error('VM cleanup failed: %s', vm.name)

    def _check_nightly_updates(self):
        """Check and perform nightly updates, killing one VM if needed."""
        # Only check once per hour
        now = datetime.now()
        if (
            self.last_update_check
            and (now - self.last_update_check).seconds < UPDATE_CHECK_INTERVAL
        ):
            return

        self.last_update_check = now

        if self.image_manager.should_update(self.config.update_hour):
            logger.info('Starting nightly image updates')

            self._make_room_for_update()

            if self.image_manager.update_runner_image():
                self.image_manager.last_update = datetime.now()
                self.image_manager._save_last_update()
                if not self.config.ephemeral:
                    self._cleanup_existing_vms()

    def _make_room_for_update(self):
        """Kill one VM to make room for update process."""
        if len(self.vms) > 0:
            vm_to_kill = next(iter(self.vms.values()))
            logger.info('Killing VM %s for nightly update', vm_to_kill.name)
            self._cleanup_vm(vm_to_kill)

    def _cleanup_existing_vms(self):
        """Destroy existing VMs to use updated runner image."""
        logger.info('Destroying existing VMs to use updated image')
        for _, vm in list(self.vms.items()):
            self._cleanup_vm(vm)


def setup_logging(
    verbose: bool = False,
    log_file: str | None = None,
    log_level: str = 'INFO',
):
    """Setup logging configuration."""
    # Determine log level
    if verbose:
        level = logging.DEBUG
    else:
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        level = level_map.get(log_level.upper(), logging.INFO)

    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def create_sample_config(config_path: str):
    """Create a sample configuration file."""
    sample_config = {
        'github_pat': 'your_github_personal_access_token',
        'organization': 'your_organization',
        'repository': 'your_repository',
        'base_image': 'ghcr.io/cirruslabs/macos-base:latest',
        'runner_image': 'your_runner_image_name',
        'ssh_username': DEFAULT_SSH_USERNAME,
        'convert_command': 'cd actions-runner && ./config.sh --help',
        'update_hour': DEFAULT_UPDATE_HOUR,
        'max_vms': MAX_VMS,
        'labels': ['macos', 'tart'],
        'ephemeral': True,
        'vm_prefix': 'tarty',
    }

    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print('Sample configuration created at: ' + config_path)
    print('Please edit the configuration file with your actual values.')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Tarty - Tart VM Manager for GitHub Runners'
    )
    parser.add_argument(
        '--config', '-c', default='tarty.json', help='Configuration file path'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Enable verbose logging'
    )
    parser.add_argument('--log-file', help='Log file path (optional)')
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Log level (default: INFO)',
    )
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create sample configuration file',
    )

    # Configuration overrides
    parser.add_argument('--github-pat', help='Override GitHub PAT')
    parser.add_argument('--organization', help='Override organization')
    parser.add_argument('--repository', help='Override repository')
    parser.add_argument('--base-image', help='Override base image')
    parser.add_argument('--runner-image', help='Override runner image')
    parser.add_argument('--ssh-username', help='Override SSH username')
    parser.add_argument('--convert-command', help='Override convert command')
    parser.add_argument(
        '--update-hour', type=int, help='Override update hour (0-23)'
    )
    parser.add_argument('--max-vms', type=int, help='Override max VMs (1-2)')
    parser.add_argument('--labels', nargs='*', help='Override labels')
    parser.add_argument('--vm-prefix', help='Override VM name prefix')
    parser.add_argument(
        '--ephemeral', action='store_true', help='Make runners ephemeral'
    )
    parser.add_argument(
        '--no-ephemeral',
        action='store_true',
        help='Make runners non-ephemeral',
    )

    args = parser.parse_args()

    if args.create_config:
        create_sample_config(args.config)
        return

    setup_logging(args.verbose, args.log_file, args.log_level)

    if not Path(args.config).exists():
        logger.error('Configuration file not found: %s', args.config)
        logger.info(
            'Run with --create-config to create a sample configuration file'
        )
        sys.exit(1)

    try:
        config = TartyConfig(args.config, args)
        orchestrator = TartyOrchestrator(config)
        orchestrator.start()
    except Exception as e:
        logger.error('Fatal error: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
