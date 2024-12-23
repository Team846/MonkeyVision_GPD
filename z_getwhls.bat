set output_dir=C:/Users/knott/Documents/FRC_CODE/VisionO24/z_whls

if not exist "%output_dir%" mkdir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310 --only-binary=:all:

echo Downloading Dash...
pip download dash --destination-dir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310 --only-binary=:all:

echo Downloading Flask...
pip download flask --destination-dir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310  --only-binary=:all:

echo Downloading NumPy...
pip download numpy --destination-dir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310 --only-binary=:all:

echo Downloading OpenCV...
pip download opencv-python --destination-dir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310 --only-binary=:all:

echo Downloading Numba...
pip download numba --destination-dir "%output_dir%" --platform manylinux2014_aarch64 --python-version 310 --only-binary=:all:

echo All downloads are complete!
