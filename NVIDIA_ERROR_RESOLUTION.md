# Résolution des erreurs d'installation NVIDIA - EmoIA

## Problème rencontré

Erreur lors de l'installation des drivers NVIDIA :
```
/tmp/apt-dpkg-install-fgxMJO/08-libnvidia-gl-535_535.247.01-0ubuntu1_amd64.deb
needrestart is being skipped since dpkg has failed
E: Sub-process /usr/bin/dpkg returned an error code (1)
```

## Solutions disponibles

### Solution 1 : Script de correction automatique (RECOMMANDÉ)

Un script de correction a été créé pour résoudre automatiquement ce problème :

```bash
sudo ./fix_nvidia_installation.sh
```

Le script propose trois options :
1. **Nettoyer et réinstaller NVIDIA** (recommandé) - Supprime proprement les installations corrompues et réinstalle
2. **Configurer pour fonctionnement CPU uniquement** - Configure EmoIA pour fonctionner sans GPU
3. **Quitter** - Pour annuler l'opération

### Solution 2 : Correction manuelle

Si vous préférez corriger manuellement :

1. **Nettoyer l'installation corrompue :**
```bash
sudo apt remove --purge nvidia-* libnvidia-*
sudo apt autoremove -y
sudo apt autoclean
```

2. **Corriger les dépendances cassées :**
```bash
sudo apt --fix-broken install -y
sudo dpkg --configure -a
```

3. **Réinstaller les drivers :**
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
```

### Solution 3 : Installation sans GPU

Si l'installation NVIDIA continue d'échouer, vous pouvez configurer EmoIA pour fonctionner en mode CPU uniquement :

```bash
# Marquer pour installation CPU uniquement
touch /tmp/emoia_cpu_only

# Puis relancer l'installation EmoIA
sudo ./install_ubuntu_server.sh
```

## Après correction

1. **Si vous avez réussi l'installation NVIDIA :**
   - Redémarrez le serveur : `sudo reboot`
   - Après redémarrage, testez : `nvidia-smi`

2. **Si vous utilisez le mode CPU :**
   - Continuez l'installation EmoIA normalement
   - Les performances seront réduites mais le système fonctionnera

3. **Reprendre l'installation EmoIA :**
```bash
sudo ./install_ubuntu_server.sh
```

## Améliorations apportées

Le script d'installation principal (`install_ubuntu_server.sh`) a été amélioré avec :

- ✅ Gestion d'erreur robuste pour NVIDIA
- ✅ Installation automatique en mode CPU si NVIDIA échoue
- ✅ Multiples méthodes d'installation des drivers
- ✅ Nettoyage automatique en cas de problème
- ✅ Installation de PyTorch appropriée (CPU/CUDA)

## Dépannage supplémentaire

### Vérifier la carte graphique
```bash
lspci | grep -i nvidia
```

### Vérifier l'installation actuelle
```bash
nvidia-smi
nvcc --version
```

### Consulter les logs d'erreur
```bash
dmesg | grep -i nvidia
journalctl -u nvidia-persistenced
```

### Problème de Secure Boot

Si Secure Boot est activé, vous devrez peut-être :
1. Désactiver Secure Boot dans le BIOS
2. Ou signer les modules NVIDIA

### Contactez-nous

Si le problème persiste, contactez l'équipe de support avec :
- La sortie de `lspci | grep -i nvidia`
- Les logs d'erreur complets
- La version d'Ubuntu : `lsb_release -a` 