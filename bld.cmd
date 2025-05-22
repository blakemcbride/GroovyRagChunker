@echo off
rem === build.cmd ===========================================================
rem Compile Tasks.java and BuildUtils.java only when the .java file
rem is newer (has a later timestamp) than its corresponding .class file.
rem Then run builder.Tasks, forwarding any arguments that were supplied
rem to this script.

setlocal enableextensions

:: --------------------------------------------------------------------------
:: Configuration – adjust if your layout changes
:: --------------------------------------------------------------------------
set "SRC_DIR=builder"
set "JAR=%SRC_DIR%\commons-compress-1.20.jar"
set "CLASSPATH=%JAR%"
set "SOURCES=%SRC_DIR%\Tasks.java %SRC_DIR%\BuildUtils.java"
set "MAIN=builder.Tasks"

:: --------------------------------------------------------------------------
:: Helper : sets NEED_COMPILE=1 if %%~t1 (file-1 time) > %%~t2 (file-2 time)
:: --------------------------------------------------------------------------
:check_newer
for %%A in ("%~1") do for %%B in ("%~2") do (
    if "%%~tA" GTR "%%~tB" set "NEED_COMPILE=1"
)
exit /b

:: --------------------------------------------------------------------------
:: Test timestamps
:: --------------------------------------------------------------------------
set "NEED_COMPILE="
call :check_newer "%SRC_DIR%\Tasks.java"      "%SRC_DIR%\Tasks.class"
call :check_newer "%SRC_DIR%\BuildUtils.java" "%SRC_DIR%\BuildUtils.class"

:: --------------------------------------------------------------------------
:: Compile when necessary
:: --------------------------------------------------------------------------
if defined NEED_COMPILE (
    echo Compiling Java sources...
    javac -cp "%CLASSPATH%" %SOURCES%
    if errorlevel 1 (
        echo Compilation failed – aborting.
        exit /b 1
    )
)

:: --------------------------------------------------------------------------
:: Run the program, passing along all original arguments
:: --------------------------------------------------------------------------
java -cp ".;%CLASSPATH%" -Dsun.security.pkcs11.enable-solaris=false %MAIN% %*

endlocal
rem ========================================================================
