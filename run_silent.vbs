' Silent runner for Windows Task Scheduler
' Executes GitAgentic in background without showing a command prompt window

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Detect Python executable
pythonExe = "C:\Users\Utkar\AppData\Local\Programs\Python\Python314\python.exe"
If Not fso.FileExists(pythonExe) Then
    pythonExe = "python.exe"
End If

runScript = scriptDir & "\main.py"

args = " --run-once"
For i = 0 To WScript.Arguments.Count - 1
    args = args & " " & WScript.Arguments(i)
Next

cmdLine = """" & pythonExe & """ """ & runScript & """" & args
WshShell.CurrentDirectory = scriptDir
WshShell.Run cmdLine, 0, False
