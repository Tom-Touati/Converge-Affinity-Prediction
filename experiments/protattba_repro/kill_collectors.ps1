# Stop every running pull_ladder collector, by PID.
#
# This is a script and not an inline `powershell -Command "..."` inside bring_up.sh for a
# concrete reason: the inline form has to survive bash double-quoting, `$_` and `$c` were
# mangled to nothing, and it cheerfully reported "stopped 0 old collectors" while leaving
# them all running. `pkill -f` had the same failure mode -- it cannot see Windows processes
# from Git Bash at all. Both reported success while doing nothing, and collectors piled up
# until sixteen were polling at once.
$c = @(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
       Where-Object { $_.CommandLine -like '*pull_ladder*' })
$c | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$left = @(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
          Where-Object { $_.CommandLine -like '*pull_ladder*' }).Count
"stopped $($c.Count) collector processes, $left remain"
