# vhs tape for the hearsay money shot: paste a URL → beautiful markdown appears.
# Regenerate the gif with: ./demo/record.sh
# Docs: https://github.com/charmbracelet/vhs

Output demo/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 18
Set TypingSpeed 60ms

Hide
Type "cd $HEARSAY_DEMO_DIR" Enter
Type "clear" Enter
Show

Type "# hearsay — turn a YouTube video into clean, timestamped markdown"
Enter
Sleep 1200ms

Type 'hearsay "https://www.youtube.com/watch?v=rStL7niR7gs" -o leader.md'
Enter
Sleep 6s

Type "head -20 leader.md"
Enter
Sleep 4s
