
#!/usr/bin/env bash

# -------------------------------
# DINOv3 Model Download Script
# -------------------------------

# Usage:
#   ./download_dinov3.sh MODEL_NAME
#
# Example:
#   ./download_dinov3.sh vit_s16
#
# Supported models:
#   vit_s16
#   vit_s_plus_16
#   vit_b16
#   vit_l16
#   vit_h_plus_16
#   vit_7b16
#
# Fill the URLs below before running.
# -------------------------------

# Model URLs (FILL THESE IN)
VIT_S16_URL=""
VIT_SPLUS16_URL=""
VIT_B16_URL="https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidjZ3OGU0NjFlMWM3dmpjeTRzY21lbXdoIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjM3NTgzOTF9fX1dfQ__&Signature=uHNjMhm%7Ep61orC65J5tGMOrENlD7I58bgDo3djozlhgPwr3B4bR5HDzf%7EEONNFPiBs5JToiLBIWalntDG%7Ed4UW6dvWvbnSSizbXuYkiOnktxzC77%7EWG5CerqMq-1zpH8eHIgsvx0uPszIssqZK65vSdwwJBnGlbease%7EvXI1Vsz9zn2qyTzgdB9ocP4vokCbO1TsI9KO7C1yzOFRwEdjG8XxqL29AT5Po7KOlhEFttyhN4b92xZ0GqBPpym9AkZDtUyRqKcdG2woDbx9LRCDCHSlpsSSGxw4aJ10SjrgyfZVmie4AZjn6SmMDIcrNaZHfaxkKJtwyAO2a8rduAFjVA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=819090567672832"
VIT_L16_URL="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidjZ3OGU0NjFlMWM3dmpjeTRzY21lbXdoIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjM3NTgzOTF9fX1dfQ__&Signature=uHNjMhm%7Ep61orC65J5tGMOrENlD7I58bgDo3djozlhgPwr3B4bR5HDzf%7EEONNFPiBs5JToiLBIWalntDG%7Ed4UW6dvWvbnSSizbXuYkiOnktxzC77%7EWG5CerqMq-1zpH8eHIgsvx0uPszIssqZK65vSdwwJBnGlbease%7EvXI1Vsz9zn2qyTzgdB9ocP4vokCbO1TsI9KO7C1yzOFRwEdjG8XxqL29AT5Po7KOlhEFttyhN4b92xZ0GqBPpym9AkZDtUyRqKcdG2woDbx9LRCDCHSlpsSSGxw4aJ10SjrgyfZVmie4AZjn6SmMDIcrNaZHfaxkKJtwyAO2a8rduAFjVA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=819090567672832"
VIT_HPLUS16_URL=""
VIT_7B16_URL=""

MODEL="$1"

if [ -z "$MODEL" ]; then
    echo "Error: No model specified."
    echo "Usage: ./download_dinov3.sh <model_name>"
    exit 1
fi

case "$MODEL" in
    vit_s16)
        URL="$VIT_S16_URL"
        OUTFILE="dinov3_vit-s16.pth"
        ;;
    vit_s_plus_16)
        URL="$VIT_SPLUS16_URL"
        OUTFILE="dinov3_vit-s+16.pth"
        ;;
    vit_b16)
        URL="$VIT_B16_URL"
        OUTFILE="dinov3_vit-b16.pth"
        ;;
    vit_l16)
        URL="$VIT_L16_URL"
        OUTFILE="dinov3_vit-l16.pth"
        ;;
    vit_h_plus_16)
        URL="$VIT_HPLUS16_URL"
        OUTFILE="dinov3_vit-h+16.pth"
        ;;
    vit_7b16)
        URL="$VIT_7B16_URL"
        OUTFILE="dinov3_vit-7b16.pth"
        ;;
    *)
        echo "Error: Unknown model '$MODEL'"
        exit 1
        ;;
esac

if [ -z "$URL" ]; then
    echo "Error: URL for $MODEL is not set. Edit the script and fill in the correct link."
    exit 1
fi

echo "Downloading $MODEL ..."
wget -O "$OUTFILE" "$URL"
echo "Saved to $OUTFILE"
