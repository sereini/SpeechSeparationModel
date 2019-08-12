# BA LipReader 2019
# author: Sereina Scherrer
# date: 20.06.2019
# install requirements for video download

#define path
$scriptFolder   = Split-Path -Parent $MyInvocation.MyCommand.Path
$test_dir = $scriptFolder + "\env_AVSpeech\Lib\"

# start
write-Host "start install a virtual environment" -ForegroundColor Yellow
pip install virtualenv

cd $scriptFolder

if(!(Test-Path -Path $test_dir)){
    Write-Host "installing virtual environment" -ForegroundColor Yellow
    virtualenv env_AVSpeech
}

.\env_AVSpeech\Scripts\activate

write-Host "virtual environment installed and activated" -ForegroundColor Yellow
write-Host "installation path: $scriptFolder" -ForegroundColor Yellow

write-Host "start install required packages" -ForegroundColor Yellow
try{
    pip install tensorflow==1.14.0
    write-Host "tensorflow installed successfully" -ForegroundColor Green

    pip install pillow
    write-Host "pillow installed successfully" -ForegroundColor Green

    pip install scikit-learn==0.21.2
    write-Host "scikit-learn installed successfully" -ForegroundColor Green

    pip install pandas==0.23.0
    write-Host "pandas installed successfully" -ForegroundColor Green

    pip install numpy==1.16.2
    write-Host "numpy installed successfully" -ForegroundColor Green

    pip install matplotlib==2.2.2
    write-Host "matplotlib installed successfully" -ForegroundColor Green

    pip install scipy==1.1.0
    write-Host "scipy installed successfully" -ForegroundColor Green


    pip install opencv-python==4.1.0.25
    write-Host "opencv installed successfully" -ForegroundColor Green

    pip install youtube_dl==2019.6.8
    write-Host "youtube_dl installed successfully" -ForegroundColor Green

    pip install pafy==0.5.4
    write-Host "pafy installed successfully" -ForegroundColor Green

    pip install termcolor==1.1.0
    write-Host "termcolor installed successfully" -ForegroundColor Green
}catch{
    write-Host "something failed" -ForegroundColor Red
}finally{
    write-Host "done." -ForegroundColor Yellow
}