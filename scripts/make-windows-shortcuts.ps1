<#
  Creates "Start/Stop Boogu UI" shortcuts on your Windows Desktop that drive the UI on a
  remote Linux host over SSH (uses your existing ~/.ssh/config, key-based).

  Example:
    powershell -File scripts\make-windows-shortcuts.ps1 -SshHost myserver -Url http://10.0.0.5:8771 -RepoDir '~/Boogu-Image'
#>
param(
  [Parameter(Mandatory=$true)][string]$SshHost,
  [Parameter(Mandatory=$true)][string]$Url,
  [string]$RepoDir = '~/Boogu-Image',
  [string]$OutDir  = [Environment]::GetFolderPath('Desktop')
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$start = Join-Path $here 'Start-Boogu-UI.cmd'
@"
@echo off
title Boogu-Image UI - START
echo Starting Boogu-Image web UI on $SshHost ...
ssh $SshHost "$RepoDir/ui.sh start"
echo Opening $Url ...
start "" "$Url"
ping -n 6 127.0.0.1 >nul
"@ | Out-File -Encoding ascii $start

$stop = Join-Path $here 'Stop-Boogu-UI.cmd'
@"
@echo off
title Boogu-Image UI - STOP
echo Stopping Boogu-Image web UI on $SshHost ...
ssh $SshHost "$RepoDir/ui.sh stop"
ping -n 3 127.0.0.1 >nul
"@ | Out-File -Encoding ascii $stop

$ws = New-Object -ComObject WScript.Shell
foreach ($pair in @(@($start,'Start Boogu UI','shell32.dll,13'), @($stop,'Stop Boogu UI','shell32.dll,27'))) {
  $sc = $ws.CreateShortcut((Join-Path $OutDir ($pair[1] + '.lnk')))
  $sc.TargetPath = $pair[0]; $sc.WorkingDirectory = $here
  $sc.IconLocation = "$env:SystemRoot\System32\$($pair[2])"; $sc.Save()
  Write-Output ("created: " + $pair[1] + '.lnk')
}
