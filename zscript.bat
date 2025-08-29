setlocal

set SOURCE_DIR=X:/Vision/MonkeyVision_GPD/*
set TARGET_USER=orangepi
set TARGET_HOST=10.8.46.204
set TARGET_DIR=/home/orangepi/MonkeyVision_GPD/

scp -P 5806 -r "%SOURCE_DIR%" "%TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%" 

echo Process completed
endlocal