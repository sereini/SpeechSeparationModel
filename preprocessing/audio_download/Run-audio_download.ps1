# BA LipReader 2019
# author: Carmen Halbeisen
# date: 08.07.2019
# run audio download from a defined start id to a defined stop id 

clear

$scriptFolder   = Split-Path -Parent $MyInvocation.MyCommand.Path
$test_dir = $scriptFolder + "\env_AVSpeech\Lib\site-packages\librosa"
#set destination of audios
$dest_dir = "$scriptFolder\audio\audio_speech\" -replace [Regex]::Escape("\"), "/"

write-Host "destination: $dest_dir"

#$start defines from which id to START the downlowad
#$max defines from which id to STOP the downlowad

$start = 5000
$max = 10000
$stop = $start

#$thread_amnt defines how many threads are used
$thread_amnt = 4
$delta_thr = [int]$($($max - $start)/$thread_amnt)

cd $scriptFolder

if(!(Test-Path -Path $test_dir)){
    write-Host "install-Packages_match.ps1 first" -ForegroundColor Red
    .\install-Packages_match.ps1 
}
write-Host "------------------------------------------------------------------------------------------------------"
.\env_AVSpeech\Scripts\activate

write-Host "virtual environment installed and activated" -ForegroundColor Yellow
write-Host "start audio download" -ForegroundColor Yellow

#starts downloading audio from id $start to id $stop and divides it between $thread_amnt threads
for ($i=0; $i -lt $thread_amnt; $i++)
{
    $start = $stop

    if($i -eq $($thread_amnt-1))
    {
        $stop = $max
    }
    else
    {
        $stop = $start + $delta_thr
    }
    Start-Process -Verb runAs cmd.exe -ArgumentList "/k python -u $scriptFolder\helperfiles\audio_download.py --dir $dest_dir --start $start --stop $stop"
    write-Host "thread: $i"
    write-Host "start id: $start"
    write-Host "stop id: $stop"
}


write-Host "download completed" -ForegroundColor Yellow
Read-Host -Prompt "Press Enter to exit"