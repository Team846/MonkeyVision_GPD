setlocal

set SOURCE_DIR=C:/Users/knott/Documents/FRC_CODE/VisionO24/*
set TARGET_USER=orangepi
set TARGET_HOST=10.8.46.7
set TARGET_DIR=/home/orangepi/MonkeyVision_AT/

scp -r "%SOURCE_DIR%" "%TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%"

echo Process completed
endlocal