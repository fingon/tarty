#!/usr/bin/env -S uv run --with pydantic python3
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
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

# Constants
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_TIMEOUT = 300
DEFAULT_SSH_USERNAME = 'admin'
VM_START_TIMEOUT = 300
ORCHESTRATION_CYCLE_INTERVAL = 30
UPDATE_CHECK_INTERVAL = 3600
MAX_VMS = 2
DEFAULT_UPDATE_HOUR = 2
RUNNER_CONFIG_TIMEOUT = 60
RUNNER_START_TIMEOUT = 30
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


# Data classes
class RunnerConfig(BaseModel):
    """Configuration for GitHub runner."""

    github_pat: str
    organization: str
    repository: str
    base_image: str
    runner_image: str
    ssh_username: str = DEFAULT_SSH_USERNAME
    convert_command: str = ''
    update_hour: int = DEFAULT_UPDATE_HOUR

    @field_validator(
        'github_pat',
        'organization',
        'repository',
        'base_image',
        'runner_image',
    )
    @classmethod
    def validate_required_fields(cls, v):
        if not v:
            raise ValueError('Field cannot be empty')
        return v

    @field_validator('update_hour')
    @classmethod
    def validate_update_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('update_hour must be between 0 and 23')
        return v


# Utility classes
class CommandRunner:
    """Utility for running subprocess commands."""

    @staticmethod
    def run_tart_command(args: list, **kwargs) -> subprocess.CompletedProcess:
        """Run a tart command with given arguments."""
        return subprocess.run(['tart'] + args, **kwargs)

    @staticmethod
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


class SSHClient:
    """SSH client for VM operations."""

    def __init__(self, host: str, username: str):
        self.host = host
        self.username = username

    def execute_command(self, command: str, timeout: int = 60) -> bool:
        """Execute a command via SSH and return success status."""
        try:
            result = CommandRunner.run_ssh_command(
                self.host, self.username, command, timeout
            )
            if result.returncode == 0:
                logging.info(f'SSH command succeeded on {self.host}')
                return True
            else:
                logging.error(
                    f'SSH command failed on {self.host}: {result.stderr}'
                )
                return False
        except Exception as e:
            logging.error(f'SSH command execution failed: {e}')
            return False

    def is_available(
        self, timeout: int = DEFAULT_SSH_TIMEOUT, retry: int = 1
    ) -> bool:
        """Check if SSH is available on the host."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                ) as sock:
                    sock.settimeout(min(5, timeout))
                    result = sock.connect_ex((self.host, DEFAULT_SSH_PORT))
                    if result == 0:
                        logging.info(
                            f'SSH available on {self.host}:{DEFAULT_SSH_PORT}'
                        )
                        return True
            except Exception:
                pass
            time.sleep(retry)

        logging.error(f'SSH timeout after {timeout} seconds')
        return False


def log_vm_operation(
    operation: str, vm_name: str, success: bool, details: str = ''
):
    """Log VM operations with consistent format."""
    level = logging.INFO if success else logging.ERROR
    status = 'succeeded' if success else 'failed'
    message = f'VM {operation} {status}: {vm_name}'
    if details:
        message += f' - {details}'
    logging.log(level, message)


class TartyConfig:
    """Configuration management for tarty."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.runner_config = self._load_config()

    def _load_config(self) -> RunnerConfig:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(
                f'Configuration file not found: {self.config_path}'
            )
        except json.JSONDecodeError as e:
            raise ConfigError(f'Invalid JSON in config file: {e}')

        try:
            return RunnerConfig(**config_data)
        except Exception as e:
            raise ConfigError(f'Invalid configuration: {e}')

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


class TartVM:
    """Represents a single tart VM instance."""

    def __init__(self, name: str, image: str):
        self.name = name
        self.image = image
        self.state = VMState.STOPPED
        self.runner_token = None
        self.process = None
        self._ip_address = None

    @property
    def running(self) -> bool:
        """Check if VM is running."""
        return self.state == VMState.RUNNING

    def start(self) -> bool:
        """Start the VM and wait for SSH to be available."""
        try:
            log_vm_operation('start', self.name, False, 'attempting')
            self.state = VMState.STARTING

            self._start_vm_process()

            # Wait for SSH to be available
            vm_ip = self.get_vm_ip()
            if not vm_ip:
                raise VMStartError(f'Failed to get IP for VM {self.name}')

            ssh_client = SSHClient(
                vm_ip, 'admin'
            )  # Will be configurable later
            if not ssh_client.is_available(VM_START_TIMEOUT):
                raise VMStartError(f'SSH not available for VM {self.name}')

            self.state = VMState.RUNNING
            self._ip_address = vm_ip
            log_vm_operation(
                'start', self.name, True, f'SSH available on {vm_ip}'
            )
            return True

        except (VMStartError, subprocess.CalledProcessError) as e:
            log_vm_operation('start', self.name, False, str(e))
            self.stop()
            return False

    def _start_vm_process(self):
        """Start the VM process in a separate thread."""

        def run_vm():
            try:
                self.process = CommandRunner.run_tart_command(
                    ['run', '--no-graphics', self.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logging.error(f'VM process failed: {e}')

        vm_thread = threading.Thread(target=run_vm, daemon=True)
        vm_thread.start()

    def get_vm_ip(self) -> Optional[str]:
        """Get the IP address of the VM."""
        if self._ip_address:
            return self._ip_address

        try:
            result = CommandRunner.run_tart_command(
                ['ip', self.name],
                capture_output=True,
                text=True,
                check=True,
            )
            self._ip_address = result.stdout.strip()
            return self._ip_address
        except subprocess.CalledProcessError as e:
            logging.error(f'Failed to get VM IP: {e}')
            return None

    def stop(self) -> bool:
        """Stop the VM."""
        try:
            if self.state == VMState.STOPPED:
                return True

            log_vm_operation('stop', self.name, False, 'attempting')
            self.state = VMState.STOPPING

            CommandRunner.run_tart_command(
                ['stop', self.name], check=True, capture_output=True
            )

            self.state = VMState.STOPPED
            self.process = None
            self._ip_address = None
            log_vm_operation('stop', self.name, True)
            return True

        except subprocess.CalledProcessError as e:
            log_vm_operation('stop', self.name, False, str(e))
            # Even if tart stop fails, we consider the VM stopped from our perspective
            # as we can't interact with it further.
            self.state = VMState.STOPPED
            self.process = None
            self._ip_address = None
            return False

    def destroy(self) -> bool:
        """Destroy the VM."""
        try:
            if self.running:
                self.stop()

            log_vm_operation('destroy', self.name, False, 'attempting')

            CommandRunner.run_tart_command(
                ['delete', self.name], check=True, capture_output=True
            )

            log_vm_operation('destroy', self.name, True)
            return True

        except subprocess.CalledProcessError as e:
            log_vm_operation('destroy', self.name, False, str(e))
            return False


class GitHubRunnerManager:
    """Manages GitHub runner registration and lifecycle."""

    def __init__(self, config: TartyConfig):
        self.config = config

    def get_registration_token(self) -> Optional[str]:
        """Get a registration token from GitHub API."""
        try:
            url = f'https://api.github.com/repos/{self.config.organization}/{self.config.repository}/actions/runners/registration-token'

            req = urllib.request.Request(url, method='POST')
            req.add_header('Authorization', f'token {self.config.github_pat}')
            req.add_header('Accept', 'application/vnd.github.v3+json')

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                token = data.get('token')
                if token:
                    logging.info(
                        'Successfully obtained GitHub registration token'
                    )
                return token

        except Exception as e:
            logging.error(f'Failed to get registration token: {e}')
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
                    f'Failed to get IP for VM {vm.name}'
                )

            ssh_client = SSHClient(vm_ip, self.config.ssh_username)

            # Configure the runner
            if not self._configure_runner(ssh_client, token, vm.name):
                raise RunnerRegistrationError('Failed to configure runner')

            # Start the runner
            if not self._start_runner(ssh_client):
                raise RunnerRegistrationError('Failed to start runner')

            vm.runner_token = token
            logging.info(f'Runner registered and started for VM: {vm.name}')
            return True

        except RunnerRegistrationError as e:
            logging.error(f'Failed to register runner for VM {vm.name}: {e}')
            return False

    def _configure_runner(
        self, ssh_client: SSHClient, token: str, runner_name: str
    ) -> bool:
        """Configure the GitHub runner via SSH."""
        repo_url = f'https://github.com/{self.config.organization}/{self.config.repository}'
        command = f'cd actions-runner && ./config.sh --url {repo_url} --token {token} --name {runner_name} --ephemeral --unattended'

        logging.info(f'Configuring runner on {ssh_client.host}')
        return ssh_client.execute_command(command, RUNNER_CONFIG_TIMEOUT)

    def _start_runner(self, ssh_client: SSHClient) -> bool:
        """Start the GitHub runner via SSH."""
        command = 'cd actions-runner && nohup ./run.sh > runner.log 2>&1 &'

        logging.info(f'Starting runner on {ssh_client.host}')
        return ssh_client.execute_command(command, RUNNER_START_TIMEOUT)


class ImageManager:
    """Manages tart image updates and creation."""

    def __init__(self, config: TartyConfig):
        self.config = config
        self.last_update = self._load_last_update()

    def _get_update_file_path(self) -> Path:
        """Get the path to the last update timestamp file."""
        return Path.home() / '.tarty_last_update.json'

    def _load_last_update(self) -> Optional[datetime]:
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
            logging.error(f'Failed to save last update time: {e}')

    def create_runner_image(self, vm_name: str) -> bool:
        """Create a runner image from the runner base image."""
        try:
            logging.info(f'Creating runner image: {vm_name}')
            CommandRunner.run_tart_command(
                ['clone', self.config.runner_image, vm_name],
                check=True,
                capture_output=True,
            )
            logging.info(f'Runner image {vm_name} created successfully')
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f'Failed to create runner image {vm_name}: {e}')
            return False

    def update_runner_image(self) -> bool:
        """Update the runner image by converting base image."""
        if not self.config.convert_command:
            logging.info(
                'No convert command specified, skipping runner image update'
            )
            return True

        temp_vm_name = f'tarty-convert-{int(time.time())}'

        try:
            logging.info('Starting runner image update process')

            temp_vm = self._create_temp_vm(temp_vm_name)
            if not temp_vm:
                return False

            if not self._run_conversion(temp_vm):
                self._cleanup_temp_vm(temp_vm_name)
                return False

            return self._finalize_image_update(temp_vm, temp_vm_name)

        except Exception as e:
            logging.error(f'Failed to update runner image: {e}')
            self._cleanup_temp_vm(temp_vm_name)
            return False

    def _create_temp_vm(self, temp_vm_name: str) -> Optional[TartVM]:
        """Create and start temporary VM for conversion."""
        try:
            # Clone base image to temporary VM
            logging.info(
                f'Cloning base image to temporary VM: {temp_vm_name}'
            )
            CommandRunner.run_tart_command(
                ['clone', self.config.base_image, temp_vm_name],
                check=True,
                capture_output=True,
            )

            # Create and start temporary VM instance
            temp_vm = TartVM(temp_vm_name, temp_vm_name)
            if not temp_vm.start():
                raise ImageUpdateError(
                    'Failed to start temporary VM for conversion'
                )

            return temp_vm

        except (subprocess.CalledProcessError, ImageUpdateError) as e:
            logging.error(f'Failed to create temporary VM: {e}')
            return None

    def _run_conversion(self, temp_vm: TartVM) -> bool:
        """Run conversion command on temporary VM."""
        vm_ip = temp_vm.get_vm_ip()
        if not vm_ip:
            logging.error('Failed to get IP for temporary VM')
            return False

        logging.info(f'Running conversion command on {vm_ip}')
        ssh_client = SSHClient(vm_ip, self.config.ssh_username)

        if not ssh_client.execute_command(
            self.config.convert_command, IMAGE_CONVERSION_TIMEOUT
        ):
            logging.error('Conversion command failed')
            return False

        logging.info('Conversion command completed successfully')
        return True

    def _finalize_image_update(
        self, temp_vm: TartVM, temp_vm_name: str
    ) -> bool:
        """Finalize the image update by stopping VM and renaming."""
        try:
            # Stop the VM
            temp_vm.stop()

            # Delete old runner image if it exists
            self._delete_old_runner_image()

            # Rename converted VM to runner image
            CommandRunner.run_tart_command(
                ['rename', temp_vm_name, self.config.runner_image],
                check=True,
                capture_output=True,
            )

            logging.info('Runner image updated successfully')
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f'Failed to finalize image update: {e}')
            return False

    def _delete_old_runner_image(self):
        """Delete old runner image if it exists."""
        try:
            CommandRunner.run_tart_command(
                ['delete', self.config.runner_image],
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            pass  # Image might not exist

    def _cleanup_temp_vm(self, temp_vm_name: str):
        """Clean up temporary VM."""
        try:
            CommandRunner.run_tart_command(
                ['delete', temp_vm_name], capture_output=True
            )
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

    def __init__(self, config_path: str):
        try:
            self.config = TartyConfig(config_path)
            self.runner_manager = GitHubRunnerManager(self.config)
            self.image_manager = ImageManager(self.config)
            self.vms: Dict[str, TartVM] = {}
            self.running = False
            self.last_update_check = None
        except ConfigError as e:
            logging.error(f'Configuration error: {e}')
            sys.exit(1)

    def start(self):
        """Start the orchestrator."""
        logging.info('Starting Tarty orchestrator')
        self.running = True

        # Main orchestration loop
        try:
            while self.running:
                self._orchestration_cycle()
                time.sleep(ORCHESTRATION_CYCLE_INTERVAL)
        except KeyboardInterrupt:
            logging.info('Received interrupt signal')
        finally:
            self.stop()

    def stop(self):
        """Stop the orchestrator and clean up."""
        logging.info('Stopping Tarty orchestrator')
        self.running = False

        # Clean up all VMs
        for vm in list(self.vms.values()):
            self._cleanup_vm(vm)

    def _orchestration_cycle(self):
        """Single cycle of the orchestration loop."""
        try:
            # Check for nightly updates first (when no VMs are running)
            self._check_nightly_updates()

            # Remove completed VMs
            self._cleanup_completed_vms()

            # Start new VMs if we have capacity
            self._start_new_vms_if_needed()

        except Exception as e:
            logging.error(f'Error in orchestration cycle: {e}')

    def _cleanup_completed_vms(self):
        """Remove completed VMs from tracking."""
        completed_vms = [vm for vm in self.vms.values() if not vm.running]
        for vm in completed_vms:
            self._cleanup_vm(vm)

    def _start_new_vms_if_needed(self):
        """Start new VMs if we have capacity."""
        while len(self.vms) < MAX_VMS:
            vm_name = f'tarty-runner-{int(time.time())}'
            if self._create_and_start_vm(vm_name):
                break  # Only create one VM per cycle

    def _create_and_start_vm(self, vm_name: str) -> bool:
        """Create and start a new VM."""
        try:
            # Create runner image
            if not self.image_manager.create_runner_image(vm_name):
                return False

            # Create VM instance
            vm = TartVM(vm_name, vm_name)

            # Start VM
            if not vm.start():
                vm.destroy()
                return False

            # Register runner
            if not self.runner_manager.register_runner(vm):
                vm.destroy()
                return False

            self.vms[vm_name] = vm
            logging.info(f'Successfully created and started VM: {vm_name}')
            return True

        except Exception as e:
            logging.error(f'Failed to create and start VM {vm_name}: {e}')
            return False

    def _cleanup_vm(self, vm: TartVM):
        """Clean up a VM."""
        log_vm_operation('cleanup', vm.name, False, 'attempting')

        # Destroy VM
        success = vm.destroy()

        # Remove from tracking
        if vm.name in self.vms:
            del self.vms[vm.name]

        log_vm_operation('cleanup', vm.name, success)

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
            logging.info('Starting nightly image updates')

            # Kill one VM if we have any running to make room for update process
            self._make_room_for_update()

            # Update runner image using conversion command (base image is manually updated)
            if self.image_manager.update_runner_image():
                self.image_manager.last_update = datetime.now()
                self.image_manager._save_last_update()

    def _make_room_for_update(self):
        """Kill one VM to make room for update process."""
        if len(self.vms) > 0:
            vm_to_kill = next(iter(self.vms.values()))
            logging.info(f'Killing VM {vm_to_kill.name} for nightly update')
            self._cleanup_vm(vm_to_kill)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('tarty.log')],
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
    }

    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f'Sample configuration created at: {config_path}')
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
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create sample configuration file',
    )

    args = parser.parse_args()

    if args.create_config:
        create_sample_config(args.config)
        return

    setup_logging(args.verbose)

    if not Path(args.config).exists():
        logging.error(f'Configuration file not found: {args.config}')
        logging.info(
            'Run with --create-config to create a sample configuration file'
        )
        sys.exit(1)

    try:
        orchestrator = TartyOrchestrator(args.config)
        orchestrator.start()
    except Exception as e:
        logging.error(f'Fatal error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
