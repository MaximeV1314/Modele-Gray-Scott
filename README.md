Je présente ici un code Python permetant de visualiser la dynamique de diffusion pour un modèle de Gray-Scott.

Il suffit simplement de modifier les variables k et F en début de code pour voir apparaître différents paternes. Attention, ces variables sont assez chaotiques mais permet à la fois de découvrir plein de paternes amusants !

F doit varier entre 0.01 et 0.1 et k entre 0.04 et 0.06.

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique". Exécuter le code. Une fois fini, ouvrir le cmd depuis le dossier initial et taper : 
ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" 
une vidéo .mp4 devrait être créée.

Amélioration à venir :
- créer une interface.
