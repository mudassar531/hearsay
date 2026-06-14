# vhs tape for the hearsay *dataset* money shot:
#   a recording → a ready-to-train TTS/STT dataset (clips + transcripts).
# Regenerate the gif with: ./demo/record_dataset.sh
# Docs: https://github.com/charmbracelet/vhs

Output demo/dataset.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 760
Set Theme "Catppuccin Mocha"
Set Padding 18
Set TypingSpeed 55ms

Hide
Type "cd $HEARSAY_DEMO_DIR" Enter
Type "clear" Enter
Show

Type "# hearsay dataset — turn a recording into a TTS/STT training set"
Enter
Sleep 1200ms

Type "hearsay dataset interview.wav --out voice-data --segment-max 3"
Enter
Sleep 8s

Type "ls voice-data"
Enter
Sleep 2200ms

Type "cat voice-data/metadata.csv"
Enter
Sleep 4s
