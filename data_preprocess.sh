# Resize all images in all directories
for d in */ ; do
    convert $d'*.png' -scale 200x200\! $d'%d_resized.png'
done
