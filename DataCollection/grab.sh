#!/bin/bash
# grab all links in links.txt
mkdir Downloads
input="links.txt"
while IFS= read -r var
do
  while IFS=' | ' read -ra line
  do
    url=${line[0]}
    name=${line[1]}
  done <<< "$var"
  mkdir Downloads/$name
  # download video
  pytube -e mp4 -p Downloads/$name/ -f $name -r 720p $url
  # location of mp4 file
  video=Downloads/$name/$name.mp4
  # output directory for images
  output=Downloads/$name
  # split videos 1 frame per second
  ffmpeg -i $video -r 1 -f image2 $output/image-%07d.png -nostdin
done < "$input"
