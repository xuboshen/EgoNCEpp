#!/bin/bash

num_process=8
small_side=288
cliplen_sec=300
max_tries=5
indir="ego4d/v2/full_scale/videos" 
outdir="ego4d/v2/down_scale/videos" 

cd $indir || exit

all_videos=$(find . -iname "*.mp4")
all_videos=( $all_videos )

process_video() {
    video="$1"
    video=$(echo $video | sed 's/^.\///')


    W=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=width "${indir}/${video}" | grep width )
    W=${W#width=}
    H=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=height "${indir}/${video}" | grep height )
    H=${H#height=}

    if [ $W -gt $H ] && [ $H -gt ${small_side} ]; then
        scale_str="-filter:v scale=-1:${small_side}"
    elif [ $H -gt $W ] && [ $W -gt ${small_side} ]; then
        scale_str="-filter:v scale=${small_side}:-1"
    else
        scale_str=""
    fi

    vidlen_sec=$( ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${indir}/${video}" )

    mkdir -p "${outdir}/${video}"

    for st_sec in $(seq 0 ${cliplen_sec} ${vidlen_sec}); do
        outfpath=${outdir}/${video}/${st_sec}.mp4
        try=0
        while [ $try -le $max_tries ]; do
            ffmpeg -y -ss ${st_sec} -i "${indir}/${video}" ${scale_str} -t ${cliplen_sec} "${outfpath}"
            try=$(( $try + 1 ))
            write_errors=$( ffprobe -v error -i "${outfpath}" )
            if [ -z "$write_errors" ]; then
                echo $outfpath written successfully in $try tries!
                break
            fi
        done
    done
    echo "Converted ${video}"
}

export -f process_video

echo "${all_videos[@]}" | tr ' ' '\n' | parallel --progress -j"$num_process" process_video