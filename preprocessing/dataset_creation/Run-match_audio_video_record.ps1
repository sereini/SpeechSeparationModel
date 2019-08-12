# BA LipReader 2019
# author: Carmen Halbeisen
# date: 26.06.2019
# match audios with embeddings, mix audios and put data into tfRecords

clear

$scriptFolder   = Split-Path -Parent $MyInvocation.MyCommand.Path
$test_dir = $scriptFolder + "\env_AVSpeech\Lib\site-packages\pyfastcopy"

#$emb_dir defines where the source embeddings are saved. 
$emb_dir = 'D:\video_all\' -replace [Regex]::Escape("\"), "/"
#$aud_dir defines where the audios are saved => USE HARDDISC!
$aud_dir = 'G:\Bachelorarbeit\audio\audio_speech\' -replace [Regex]::Escape("\"), "/"
#$aud_dir defines where all (.wav, .npy, .tfrecord) files are saved. 
$dest_dir = 'D:\BA_LipReader\data_matched\'  -replace [Regex]::Escape("\"), "/"

#for copying files:
$src_dir_copy = $dest_dir
$dest_dir_copy = 'F:\BA_LipRead\'  -replace [Regex]::Escape("\"), "/"

#stop ids (10000 steps between ids) to match
#start id = stop id - 10000
$stop = @(690000)

write-Host "destination: $dest_dir"

cd $scriptFolder

if(!(Test-Path -Path $test_dir)){
    write-Host "install-Packages_match.ps1 first" -ForegroundColor Red
    .\install-Packages_match.ps1 
}
write-Host "------------------------------------------------------------------------------------------------------"
.\env_AVSpeech\Scripts\activate

write-Host "virtual environment installed and activated" -ForegroundColor Yellow
write-Host "start audio download" -ForegroundColor Yellow

for ($i=0; $i -lt $stop.length; $i++)
{
    $start = $stop[$i] - 10000
    write-Host $start
    write-Host $stop[$i]
    #start matching audios with embeddings, mix audios and put data into tfRecords
    python -u $scriptFolder\helperfiles\match_audio_video_record.py --emb_dir $emb_dir --aud_dir $aud_dir --dest_dir $dest_dir --dest_dir_copy $dest_dir_copy --start $start --stop $stop[$i]
    #copy files from source to destintation and remove them afterwards from source
    python -u $scriptFolder\helperfiles\move_files_multith.py --src_dir $src_dir_copy --dest_dir $dest_dir_copy
    write-Host $stop[$i] is done!
}