#!/usr/bin/env python3
"""
Vision Module Toggler

Manages vision modules based on SHM toggle flags using SHM watchers.
This script runs as a background daemon and starts/stops vision modules
in response to SHM flag changes without polling.
"""

import os
import sys
import signal
import subprocess
from dataclasses import dataclass
from collections import defaultdict
from threading import Event, Thread
from typing import Set, Dict, Optional

import shm
from misc.utils import register_exit_signals


@dataclass
class ModuleProcess:
    """Represents a running vision module process"""

    name: str
    source: str
    process: subprocess.Popen
    log_name: str


class ModuleToggler:
    """Manages vision modules based on SHM toggle flags"""

    def __init__(self):
        print("[INFO] Initializing ModuleToggler")
        self.running_processes: Dict[str, ModuleProcess] = {}

        # Get vehicle configuration for available modules
        try:
            from conf import vehicle

            self.vehicle_config = vehicle.vision_modules
            print(
                f"[INFO] Loaded vehicle config with modules: {list(self.vehicle_config.keys())}"
            )
        except Exception as e:
            print(f"[ERROR] Error loading vehicle configuration: {e}")
            self.vehicle_config = {}

        print("[INFO] ModuleToggler initialization complete")

    def _get_process_key(self, module_name: str, source: str) -> str:
        """Generate unique key for module+source combination"""
        return f"{module_name}_{source}"

    def _get_shm_flag_name(self, module_name: str, source: str) -> str:
        """Generate SHM flag name for module+source combination"""
        return f"{module_name}_on_{source}".replace("-", "_")

    def _is_module_enabled(self, module_name: str, source: str) -> bool:
        """Check if a specific module+source combination should be running"""
        flag_name = self._get_shm_flag_name(module_name, source)

        try:
            # Check if the specific toggle flag exists and is True
            flag_value = getattr(shm.vision_modules, flag_name, None)
            if flag_value is not None:
                enabled = bool(flag_value.get())
                return enabled
            else:
                return False
        except Exception as e:
            print(f"[ERROR] Error reading SHM flag {flag_name}: {e}")
            return False

    def _start_module(self, module_name: str, source: str) -> bool:
        """Start a vision module with specified source"""
        process_key = self._get_process_key(module_name, source)

        # Don't start if already running
        if process_key in self.running_processes:
            return True

        log_name = f"{module_name}@{source}-module"

        try:
            # Build command similar to startm.sh
            module_path = f"{os.environ.get('CUAUV_SOFTWARE', '.')}/vision/modules/{module_name}.py"
            cmd = [sys.executable, module_path, source]

            # Check if module file exists
            if not os.path.exists(module_path):
                print(f"[ERROR] Module file not found: {module_path}")
                return False

            # Set up logging
            log_dir = os.environ.get("CUAUV_LOG", "/tmp") + "/current"
            os.makedirs(log_dir, exist_ok=True)
            log_file = open(f"{log_dir}/{log_name}.log", "a")

            timestamp = subprocess.run(
                ["date", "-u", "+%Y/%m/%d %H:%M:%S UTC"], capture_output=True, text=True
            ).stdout.strip()
            log_file.write(f"Starting {' '.join(cmd)} at {timestamp}\n")
            log_file.flush()

            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group for clean shutdown
            )

            # Store process info
            self.running_processes[process_key] = ModuleProcess(
                name=module_name, source=source, process=process, log_name=log_name
            )

            print(f"[INFO] Started: {module_name} {source}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to start {module_name} {source}: {e}")
            return False

    def _stop_module(self, module_name: str, source: str) -> bool:
        """Stop a running vision module"""
        process_key = self._get_process_key(module_name, source)

        if process_key not in self.running_processes:
            return True  # Already stopped

        module_proc = self.running_processes[process_key]

        try:
            # Graceful shutdown first
            os.killpg(os.getpgid(module_proc.process.pid), signal.SIGTERM)

            # Wait a bit for graceful shutdown
            try:
                module_proc.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(module_proc.process.pid), signal.SIGKILL)
                module_proc.process.wait(timeout=2)

            print(f"[INFO] Stopped: {module_name} {source}")
            del self.running_processes[process_key]
            return True

        except Exception as e:
            print(f"[ERROR] Error stopping {module_name} {source}: {e}")
            # Remove from tracking even if stop failed
            del self.running_processes[process_key]
            return False

    def _stop_all_modules(self):
        """Stop all running modules"""
        print("[INFO] Stopping all vision modules")
        for process_key in list(self.running_processes.keys()):
            module_proc = self.running_processes[process_key]
            self._stop_module(module_proc.name, module_proc.source)

    def _cleanup_dead_processes(self):
        """Remove dead processes from tracking"""
        dead_processes = []

        for process_key, module_proc in self.running_processes.items():
            if module_proc.process.poll() is not None:
                return_code = module_proc.process.returncode
                print(
                    f"[WARN] {module_proc.name} {module_proc.source} died unexpectedly (return code: {return_code})"
                )
                dead_processes.append(process_key)

        for process_key in dead_processes:
            del self.running_processes[process_key]

    def _sync_modules_to_shm(self):
        """Synchronize running modules with current SHM state"""
        self._cleanup_dead_processes()

        # Determine what should be running based on SHM flags
        should_run: Set[str] = set()

        for module_name, config in self.vehicle_config.items():
            capture_sources = list(config.get("capture_sources", []))
            for source in capture_sources:
                if self._is_module_enabled(module_name, source):
                    process_key = self._get_process_key(module_name, source)
                    should_run.add(process_key)

        # Determine what is currently running
        currently_running = set(self.running_processes.keys())

        # Start modules that should be running but aren't
        to_start = should_run - currently_running
        to_stop = currently_running - should_run

        for process_key in to_start:
            parts = process_key.rsplit("_", 1)
            if len(parts) == 2:
                module_name, source = parts
                self._start_module(module_name, source)
            else:
                print(f"[ERROR] Invalid process key format: {process_key}")

        # Stop modules that are running but shouldn't be
        for process_key in to_stop:
            if process_key in self.running_processes:
                module_proc = self.running_processes[process_key]
                self._stop_module(module_proc.name, module_proc.source)

    def run(self, watcher, quit_event):
        """Main event loop using SHM watchers - back to working watch_thread_wrapper pattern"""
        print("[INFO] Vision module toggler ready, waiting for SHM changes...")

        # Watch the vision_modules SHM group for changes
        watcher.watch(shm.vision_modules)

        # Initial sync on startup
        self._sync_modules_to_shm()

        # Main event loop - same pattern as control system
        while not quit_event.is_set():
            try:
                # Wait for SHM changes - same pattern as auv_controld3.py
                watcher.wait()

                if quit_event.is_set():
                    break

                print("[INFO] SHM change detected, updating modules...")

                # Sync modules when SHM changes
                self._sync_modules_to_shm()

            except Exception as e:
                print(f"[ERROR] Error in main loop: {e}")

        print("[INFO] Vision module toggler shutting down")
        self._stop_all_modules()


def main():
    """Entry point using watch_thread_wrapper like control system"""
    print("[INFO] Vision Module Toggler starting up")

    toggler = ModuleToggler()

    def run_toggler(watcher, quit_event):
        """Wrapper function for watch_thread_wrapper"""
        toggler.run(watcher, quit_event)

    from misc.utils import watch_thread_wrapper

    watch_thread_wrapper(run_toggler)


if __name__ == "__main__":
    main()
