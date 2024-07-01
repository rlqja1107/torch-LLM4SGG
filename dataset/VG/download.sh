if [ $# -eq 0 ]; then
  echo "Please provide the destination folder as an argument."
  exit 1
fi

dest_folder="$1"
echo $dest_folder
cd "$dest_folder"

## Part 1
#wget -N https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
#unzip images.zip
#rm -rf images.zip

## Part 2
#wget -N https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
#unzip images2.zip
#mv images2/* images
#rm -rf images2.zip

# Meta data
# From google drive
if ! test -f "image_data.json"; then
    wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kl7Opx3niinxSW9ZePoHknOznqrSdWAE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kl7Opx3niinxSW9ZePoHknOznqrSdWAE" -O image_data.json && rm -rf ~/cookies.txt
fi

if ! test -f "VG-SGG-dicts-with-attri.json"; then
    wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=156cp5uPAjarosLZZeshHz3evIGUW8vLw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=156cp5uPAjarosLZZeshHz3evIGUW8vLw" -O VG-SGG-dicts-with-attri.json && rm -rf ~/cookies.txt
fi

# Large size
if ! test -f "VG-SGG-with-attri.h5"; then
    wget -N https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/VG-SGG-with-attri.h5
fi
