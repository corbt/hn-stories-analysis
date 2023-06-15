#! /bin/bash

set -e

# Install deps
apt-get install -y zsh curl git tmux htop

# Change the default shell to zsh
chsh -s $(which zsh)

# Copy the .zshrc file
cp .zshrc ~/.zshrc

# Install Oh My ZSH
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install ASDF version manager
# git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.12.0

# Copy the .tool-versions file
cp .tool-versions ~/.tool-versions

# Add ASDF to zsh and bash
# echo -e '\n. $HOME/.asdf/asdf.bash' >> ~/.bashrc
# echo -e '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc

# Add the nodejs plugin to ASDF
asdf plugin-add nodejs

# Install node v20.2.0
asdf install nodejs 20.2.0

# Set my git username and email
git config --global user.name "Kyle Corbitt"
git config --global user.email "kyle@corbt.com"