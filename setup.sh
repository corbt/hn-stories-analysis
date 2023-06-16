#! /bin/bash

set -e
cd $(dirname $0)

# Install deps
apt-get install -y zsh curl git tmux htop entr

# Change the default shell to zsh
chsh -s $(which zsh)

# Install Oh My ZSH
# Check whether the ~/.oh-my-zsh directory already exists
if [ ! -d "$HOME/.oh-my-zsh" ]; then
  # If not, install it
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

# Copy the .zshrc file
cp templates/.zshrc ~/.zshrc

# Install ASDF version manager
if [ ! -d "$HOME/.asdf" ]; then
  git clone https://github.com/asdf-vm/asdf.git $HOME/.asdf --branch v0.12.0

  # Add ASDF to bash
  echo -e '\n. $HOME/.asdf/asdf.sh' >> ~/.bashrc
  echo -e '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc

  . $HOME/.asdf/asdf.sh
fi

# Copy the ASDF config file
cp templates/.tool-versions templates/.default-npm-packages $HOME

# Add the nodejs plugin to ASDF
asdf plugin-add nodejs

# Install node v20.2.0
asdf install nodejs 20.2.0

# Set my git username and email
git config --global user.name "Kyle Corbitt"
git config --global user.email "kyle@corbt.com"