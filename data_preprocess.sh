# Resize all images in all directories
for d in */ ; do
    for f in $d*.png; do
        convert $f -scale 300x300\! ${f//".png"/"_resized.png"}
    done
done
