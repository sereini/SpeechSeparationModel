# BA LipReader 2019
# author: Carmen Halbeisen
# date: 20.07.2019
# install requirments for audio download

#define path
$scriptFolder   = Split-Path -Parent $MyInvocation.MyCommand.Path

# start
write-Host "start install an virtual environment" -ForegroundColor Yellow
pip install virtualenv

cd $scriptFolder
virtualenv env_AVSpeech

.\env_AVSpeech\Scripts\activate

write-Host "virtual environment installed and activated" -ForegroundColor Yellow
write-Host "installation path $path" -ForegroundColor Yellow

write-Host "start install required packages" -ForegroundColor Yellow
try{
    pip install ffmpeg
    write-Host "ffmpeg installed successfully" -ForegroundColor Green

    pip install pandas==0.23.0
    write-Host "pandas installed successfully" -ForegroundColor Green

    pip install scipy==1.1.0
    write-Host "scipy installed successfully" -ForegroundColor Green

    pip install matplotlib==2.2.2
    write-Host "matplotlib installed successfully" -ForegroundColor Green

    pip install numpy==1.16.2
    write-Host "numpy installed successfully" -ForegroundColor Green

    pip install wave
    write-Host "numpy installed successfully" -ForegroundColor Green

    pip install youtube_dl==2019.6.8
    write-Host "youtube_dl installed successfully" -ForegroundColor Green

    pip install pafy==0.5.4
    write-Host "pafy installed successfully" -ForegroundColor Green

    pip install pathlib
    write-Host "pathlib installed successfully" -ForegroundColor Green

    pip install psutil
    write-Host "psutil installed successfully" -ForegroundColor Green
	
	pip install tensorflow
    write-Host "tensorflow installed successfully" -ForegroundColor Green

    pip install pyfastcopy
    write-Host "pyfastcopy installed successfully" -ForegroundColor Green
	
	pip install librosa
    write-Host "librosa installed successfully" -ForegroundColor Green
}catch{
    write-Host "something failed" -ForegroundColor Red
}finally{
    write-Host "done." -ForegroundColor Yellow
}