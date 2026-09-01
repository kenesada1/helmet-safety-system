$ErrorActionPreference = "Stop"

Add-Type @"
using System.Runtime.InteropServices;

public static class E8SleepControl
{
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$request = $continuous -bor $systemRequired

try {
    if ([E8SleepControl]::SetThreadExecutionState($request) -eq 0) {
        throw "SetThreadExecutionState failed"
    }
    while ($true) {
        Start-Sleep -Seconds 30
        if ([E8SleepControl]::SetThreadExecutionState($request) -eq 0) {
            throw "SetThreadExecutionState refresh failed"
        }
    }
}
finally {
    [void][E8SleepControl]::SetThreadExecutionState($continuous)
}
