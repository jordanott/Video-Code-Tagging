#!/bin/bash
# grab all links in links.txt
mkdir Downloads
input="links.txt"
while IFS= read -r var
do
  while IFS=' | ' read -ra line
  do
    url=${line[0]}
    name=${line[@]:1}
  done <<< "$var"
  # download video
  {
    path=Downloads/$name"_720"
    mkdir "$path"
    pytube -e mp4 -p "$path"/ -f "$name" -r 720p $url
  }||
  {
    rm -rf "$path"
    path=Downloads/$name"_480"
    mkdir "$path"
    pytube -e mp4 -p "$path"/ -f "$name" -r 480p $url
  }||
  {
    rm -rf "$path"
    path=Downloads/$name"_360"
    mkdir "$path"
    pytube -e mp4 -p "$path"/ -f "$name" -r 360p $url
  }||
  {
    rm -rf "$path"
    echo $url >> error.txt
  }
  # location of mp4 file
  video="$path/$name.mp4"
  # split videos 1 frame per second
  ffmpeg -i "$video" -r 1 -f image2 "$path"/%d.png -nostdin
done < "$input"
