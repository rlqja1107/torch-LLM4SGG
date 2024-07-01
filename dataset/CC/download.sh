
if [ $# -eq 0 ]; then
  echo "Please provide the destination folder as an argument."
  exit 1
fi

dest_folder="$1"
echo $dest_folder
cd "$dest_folder"

if ! test -f "cc_triplet_labels.npy"; then
  wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wxxf5o3hfFKLqjAfvNlP4APWfDShE2e_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wxxf5o3hfFKLqjAfvNlP4APWfDShE2e_" -O cc_triplet_labels.npy && rm -rf ~/cookies.txt
fi

wget -N https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/Train_GCC-training.tsv


if ! test -f "cc_meta_information.json"; then
  wget -N --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WXGF76JfG7BTAhasqtt6RN_dLmmkgYCm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WXGF76JfG7BTAhasqtt6RN_dLmmkgYCm" -O cc_meta_information.json && rm -rf ~/cookies.txt
fi

python download_img.py .