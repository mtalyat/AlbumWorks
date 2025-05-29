@echo off

:: Change to the script's directory
cd /d "%~dp0"

:: set the names of the files to include (just the names)
set FILE1=albumworks.py
set FILE2=albumworks.bat
set FILE3=README.md
set FILE4=notes.txt

:: set the output zip file name
set ZIPNAME=AlbumWorks.zip

:: create the zip archive with the files using tar
tar -a -c -f "%ZIPNAME%" "%FILE1%" "%FILE2%" "%FILE3%" "%FILE4%"

if %ERRORLEVEL% neq 0 (
    echo Failed to create archive.
    exit /b %ERRORLEVEL%
)

echo Archive created: %ZIPNAME%