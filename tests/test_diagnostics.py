"""Tests for application diagnostic metadata."""

from unittest.mock import Mock, patch

from iqc_dashboard.app import (
    collect_git_diagnostic_info,
    collect_host_diagnostic_info,
    format_byte_size,
    format_diagnostic_section,
    get_cpu_model,
)


def test_format_byte_size_uses_binary_units():
    assert format_byte_size(0) == "0 B"
    assert format_byte_size(1024) == "1.00 KiB"
    assert format_byte_size(5 * 1024**3) == "5.00 GiB"
    assert format_byte_size(None) == "Unavailable"


def test_collect_git_diagnostic_info_reports_revision_and_dirty_state():
    command_results = {
        ("rev-parse", "--short=12", "HEAD"): "abc123def456",
        ("branch", "--show-current"): "main",
        ("status", "--porcelain"): " M iqc_dashboard/app.py",
        ("show", "-s", "--format=%s", "HEAD"): "Add diagnostics",
        ("show", "-s", "--format=%cI", "HEAD"): "2026-06-12T10:00:00-05:00",
    }

    with patch(
        "iqc_dashboard.app.run_git_diagnostic_command",
        side_effect=lambda *args: command_results.get(args),
    ):
        diagnostics = collect_git_diagnostic_info()

    assert diagnostics["Commit"] == "abc123def456"
    assert diagnostics["Branch"] == "main"
    assert diagnostics["Commit subject"] == "Add diagnostics"
    assert diagnostics["Working tree"] == "Dirty"


def test_collect_host_diagnostic_info_uses_psutil_when_available():
    memory = Mock(total=16 * 1024**3, available=10 * 1024**3)
    process = Mock()
    process.memory_info.return_value = Mock(rss=256 * 1024**2)
    psutil_mock = Mock()
    psutil_mock.cpu_count.side_effect = lambda logical: 8 if logical else 4
    psutil_mock.virtual_memory.return_value = memory
    psutil_mock.Process.return_value = process

    with (
        patch.dict("sys.modules", {"psutil": psutil_mock}),
        patch("iqc_dashboard.app.get_cpu_model", return_value="Test CPU"),
        patch("iqc_dashboard.app.socket.gethostname", return_value="test-host"),
    ):
        diagnostics = collect_host_diagnostic_info()

    assert diagnostics["Hostname"] == "test-host"
    assert diagnostics["CPU"] == "Test CPU"
    assert diagnostics["CPU cores"] == "4 physical / 8 logical"
    assert diagnostics["Total memory"] == "16.00 GiB"
    assert diagnostics["Available memory"] == "10.00 GiB"
    assert diagnostics["Dashboard process memory"] == "256.00 MiB"


def test_get_cpu_model_uses_macos_chip_when_sysctl_is_unavailable():
    sysctl_failure = OSError("sysctl unavailable")
    profiler_result = Mock(stdout="Hardware:\n    Chip: Apple M1 Max\n")

    with (
        patch("iqc_dashboard.app.platform.system", return_value="Darwin"),
        patch("iqc_dashboard.app.platform.processor", return_value="arm"),
        patch(
            "iqc_dashboard.app.subprocess.run",
            side_effect=[sysctl_failure, profiler_result],
        ),
    ):
        assert get_cpu_model() == "Apple M1 Max"


def test_format_diagnostic_section_preserves_order():
    formatted = format_diagnostic_section({"Commit": "abc123", "Branch": "main"})
    assert formatted == "Commit: abc123\nBranch: main"
