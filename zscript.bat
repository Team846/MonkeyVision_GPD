setlocal

set SOURCE_DIR=X:/Vision/MonkeyVision_GPD/*
set TARGET_USER=orangepi
set TARGET_HOST=funkyvision2.local
set TARGET_DIR=/home/orangepi/MonkeyVision_GPD/

scp -r "%SOURCE_DIR%" "%TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%"

echo Process completed
endlocal