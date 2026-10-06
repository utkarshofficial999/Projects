<#
.SYNOPSIS
    Registers or manages Windows Scheduled Tasks for GitAgentic Autonomous Daily Commit System.

.PARAMETER Action
    Install, Uninstall, List, RunNow, Logs (default: Install)

.PARAMETER Time
    24-hour time to trigger daily (default: '11:00')
#>

param (
    [string]$Action = "Install",
    [string]$Time = "11:00"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = "GitAgentic_Daily_Commit"
$VbsPath = Join-Path $ScriptDir "run_silent.vbs"
$LogPath = Join-Path $ScriptDir "logs\git_agentic.log"

switch ($Action.ToLower()) {
    "uninstall" {
        Write-Host "--- Removing GitAgentic Scheduled Task ---" -ForegroundColor Cyan
        $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($Existing) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Removed task: $TaskName" -ForegroundColor Green
        } else {
            Write-Host "No active GitAgentic task found." -ForegroundColor Yellow
        }
    }

    "list" {
        Write-Host "--- GitAgentic Scheduled Task Status ---" -ForegroundColor Cyan
        $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($Task) {
            $Task | Select-Object TaskName, State | Format-Table -AutoSize
            $Info = Get-ScheduledTaskInfo -TaskName $TaskName
            Write-Host "[$TaskName] Next Run: $($Info.NextRunTime) | Last Run: $($Info.LastRunTime) (Result: $($Info.LastTaskResult))" -ForegroundColor Gray
        } else {
            Write-Host "Task '$TaskName' is not registered." -ForegroundColor Yellow
        }
    }

    "runnow" {
        Write-Host "--- Triggering Immediate Milestone Run ---" -ForegroundColor Cyan
        $PyExe = "python.exe"
        $RunnerPy = Join-Path $ScriptDir "main.py"
        & $PyExe $RunnerPy --run-once
    }

    "logs" {
        Write-Host "--- Recent GitAgentic Execution Logs ---" -ForegroundColor Cyan
        if (Test-Path $LogPath) {
            Get-Content -Path $LogPath -Tail 40
        } else {
            Write-Host "No log file found at $LogPath" -ForegroundColor Yellow
        }
    }

    default {
        Write-Host "==========================================================" -ForegroundColor Magenta
        Write-Host "   Registering GitAgentic Daily Autonomous Commit Task    " -ForegroundColor Magenta
        Write-Host "==========================================================" -ForegroundColor Magenta
        Write-Host "Working Directory : $ScriptDir" -ForegroundColor Gray
        Write-Host "Daily Run Time    : $Time" -ForegroundColor Yellow

        # 1. Clean existing task if already present
        $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($Existing) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "  [-] Cleared previous task registration" -ForegroundColor DarkGray
        }

        # 2. Register new task
        $ActionObj = New-ScheduledTaskAction `
            -Execute "wscript.exe" `
            -Argument "`"$VbsPath`"" `
            -WorkingDirectory $ScriptDir

        $TriggerObj = New-ScheduledTaskTrigger -Daily -At $Time

        $SettingsObj = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1)

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $ActionObj `
            -Trigger $TriggerObj `
            -Settings $SettingsObj `
            -Force | Out-Null

        Write-Host "  [+] Registered: $TaskName -> Daily at $Time" -ForegroundColor Green
        Write-Host "  [+] 'StartWhenAvailable' enabled: If PC was off, runs immediately when powered on!" -ForegroundColor Cyan
        Write-Host "`nTask successfully installed and ready." -ForegroundColor Green
    }
}
