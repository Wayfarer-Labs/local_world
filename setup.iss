#define MyAppName "MyApp"
#define MyAppVersion "0.1.0"

[Setup]
AppId={{9E2B9D83-2E7D-4E03-9F64-5F7E3C97F5F1}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\{#MyAppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename={#MyAppName}-Setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "installer\pixi.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "pyproject.toml";     DestDir: "{app}"; Flags: ignoreversion
Source: "pixi.lock";          DestDir: "{app}"; Flags: ignoreversion
Source: "src\*";              DestDir: "{app}\src"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";
Filename: "{app}\pixi.exe";
Parameters: "run client --manifest-path pyproject.toml";
WorkingDir: "{app}"

[Run]
; Install exactly from the shipped lockfile (don’t update it). :contentReference[oaicite:4]{index=4}
Filename: "{app}\pixi.exe";
Parameters: "install --frozen --manifest-path pyproject.toml";
WorkingDir: "{app}";
Flags: waituntilterminated runhidden;
StatusMsg: "Installing Python environment (pixi)…"

; Optional “launch at end”
Filename: "{app}\pixi.exe";
Parameters: "run client --manifest-path pyproject.toml";
WorkingDir: "{app}";
Flags: nowait postinstall skipifsilent
