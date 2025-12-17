#define MyAppName "LocalWorld"
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
Source: "installer\git\*";    DestDir: "{app}\git"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "installer\pixi.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "pixi.toml";          DestDir: "{app}"; Flags: ignoreversion
Source: "src\*";              DestDir: "{app}\src"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\pixi.exe"; Parameters: "run client --manifest-path ""{app}\pixi.toml"""; WorkingDir: "{app}"

[Run]
Filename: "{cmd}"; Parameters: "/C ""set PATH={app}\git\cmd;{app}\git\mingw64\bin;%PATH% & """"{app}\pixi.exe"""" install --manifest-path """"{app}\pixi.toml"""""""; WorkingDir: "{app}"; Flags: waituntilterminated runhidden; StatusMsg: "Installing Python environment (pixi)…"
Filename: "{app}\pixi.exe"; Parameters: "run client --manifest-path ""{app}\pixi.toml"""; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
