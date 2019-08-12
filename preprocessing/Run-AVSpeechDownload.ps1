# BA LipReader 2019
# author: Sereina Scherrer
# date: 20.06.2019
# run video download

clear

$scriptFolder   = Split-Path -Parent $MyInvocation.MyCommand.Path
$test_dir = $scriptFolder + "\env_AVSpeech\Lib\site-packages\cv2"
cd $scriptFolder

if(!(Test-Path -Path $test_dir)){
    write-Host "install-Packages.ps1 first" -ForegroundColor Red
    .\install-Packages.ps1 
}
write-Host "------------------------------------------------------------------------------------------------------"
.\env_AVSpeech\Scripts\activate

write-Host "virtual environment installed and activated" -ForegroundColor Yellow
write-Host "start download" -ForegroundColor Yellow


python -u video_download.py --start_id 0 --stop_id 10 --data_dir "images_AVSpeech"

write-Host "download completed" -ForegroundColor Yellow
Read-Host -Prompt "Press Enter to exit"
