import os
from sys import executable as pyexecutable
import subprocess
import pathlib
import gc

def Gitclone(URI:str,ClonePath:str = "") -> int :
  if(ClonePath == "") :
    while True:
      i=subprocess.run([r"git",r"clone",URI])
      if(i.returncode == 0 ): 
       del i
       gc.collect()
       return 0
      else :
       del i
  else: 
    while True:
      i=subprocess.run([r"git",r"clone",URI,ClonePath])
      if(i.returncode == 0 ): 
       del i
       gc.collect()
       return 0
      else :
       del i
def DownLoad(URI:str,DownloadPath:str,DownLoadFileName:str ) -> int:
  while (True):
    i=subprocess.run([r"aria2c",r"-c",r"-x" ,r"16", r"-s",r"16", r"-k" ,r"1M" ,r"-m",r"0",r"--enable-mmap=false",r"--console-log-level=error",r"-d",DownloadPath,r"-o",DownLoadFileName,URI]);
    if(i.returncode == 0 ): 
      del i
      gc.collect()
      return 0
    else :
      del i
user_home =pathlib.Path.home().resolve()
os.chdir(str(user_home))
#clone stable-diffusion-webui repo
print("cloning stable-diffusion-webui repo")
Gitclone(r"https://github.com/AUTOMATIC1111/stable-diffusion-webui.git",str(user_home / r"stable-diffusion-webui"))
os.chdir(str(user_home / r"stable-diffusion-webui"))
os.system("git reset --hard 89f9faa63388756314e8a1d96cf86bf5e0663045")
#

#install extensions
print("installing extensions")
Gitclone(r"https://huggingface.co/embed/negative",str(user_home / r"stable-diffusion-webui" / r"embeddings"  / r"negative"))
Gitclone(r"https://huggingface.co/embed/lora",str(user_home / r"stable-diffusion-webui" / r"models" / r"Lora" / r"positive"))
DownLoad(r"https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth",str(user_home / r"stable-diffusion-webui" / r"models" / r"ESRGAN") ,r"4x-UltraSharp.pth")
while True: 
  if(subprocess.run([r"wget",r"https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py",r"-O",str(user_home / r"stable-diffusion-webui" / r"scripts" / r"run_n_times.py")]).returncode == 0):
    break
Gitclone(r"https://github.com/deforum-art/deforum-for-automatic1111-webui",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"deforum-for-automatic1111-webui" ))
Gitclone(r"https://github.com/AlUlkesh/stable-diffusion-webui-images-browser",str(user_home / r"stable-diffusion-webui" / r"extensions"/ r"stable-diffusion-webui-images-browser"))
Gitclone(r"https://github.com/camenduru/stable-diffusion-webui-huggingface",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-huggingface"))
Gitclone(r"https://github.com/camenduru/sd-civitai-browser",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-civitai-browser"))
Gitclone(r"https://github.com/kohya-ss/sd-webui-additional-networks",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks"))
Gitclone(r"https://github.com/Mikubill/sd-webui-controlnet",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-controlnet"))
Gitclone(r"https://github.com/fkunn1326/openpose-editor",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"openpose-editor"))
Gitclone(r"https://github.com/jexom/sd-webui-depth-lib",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-depth-lib"))
Gitclone(r"https://github.com/hnmr293/posex",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"posex"))
Gitclone(r"https://github.com/nonnonstop/sd-webui-3d-open-pose-editor",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-3d-open-pose-editor"))
#中文本地化的请解除下一行的注释
#Gitclone(r"https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN.git",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-localization-zh_CN"))
Gitclone(r"https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git" , str(user_home / r"stable-diffusion-webui" / r"extensions" / r"a1111-sd-webui-tagcomplete"))
Gitclone(r"https://github.com/camenduru/sd-webui-tunnels",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-tunnels"))
Gitclone(r"https://github.com/etherealxx/batchlinks-webui",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"batchlinks-webui"))
Gitclone(r"https://github.com/catppuccin/stable-diffusion-webui",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-catppuccin"))

#Gitclone(r"https://github.com/KohakuBueleaf/a1111-sd-webui-locon",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"a1111-sd-webui-locon" ))
Gitclone(r"https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-rembg"))
Gitclone(r"https://github.com/ashen-sensored/stable-diffusion-webui-two-shot",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"stable-diffusion-webui-two-shot"))
Gitclone(r"https://github.com/camenduru/sd_webui_stealth_pnginfo",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd_webui_stealth_pnginfo"))

os.chdir(user_home / r"stable-diffusion-webui")

#download ControlNet models
print("extensions dolwnload done .\ndownloading ControlNet models")
dList =[r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_seg_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd14v1.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth",
               r"https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth"]
for i in range(0,len(dList)): DownLoad(dList[i],str(user_home / "stable-diffusion-webui" / "extensions" / "sd-webui-controlnet" / "models"),pathlib.Path(dList[i]).name)
del dList

#download model    
#you can change model download address here
print("ControlNet models download done.\ndownloading model")
DownLoad(r"https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.5-pruned.ckpt",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"anything-v4.5-pruned.ckpt")
DownLoad(r"https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.0.vae.pt",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"anything-v4.0.vae.pt")
DownLoad(r"https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fp16.safetensors",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"Counterfeit-V3.0_fp16.safetensors")
DownLoad(r"https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A1B_orangemixs.safetensors",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"AOM3A1B_orangemixs.safetensors")
DownLoad(r"https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"orangemix.vae.pt")
DownLoad(r"https://huggingface.co/Meina/MeinaPastel/resolve/main/MeinaPastelV5%20-%20Baked%20VAE.safetensors",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"MeinaPastelV5_BakedVAE.safetensors")
DownLoad(r"https://huggingface.co/Meina/MeinaPastel/resolve/main/MeinaPastelV5%20-%20Without%20VAE.safetensors",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"MeinaPastelV5_WithoutVAE.safetensors")
DownLoad(r"https://civitai.com/api/download/models/9474",str(user_home / r"stable-diffusion-webui" / r"models" / r"Stable-diffusion"),r"chilloutmix_NiPrunedFp16.safetensors")

DownLoad(r"https://civitai.com/api/download/models/39885",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora"),r"Better_light.safetensors")
DownLoad(r"https://civitai.com/api/download/models/21065",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora"),r"LAS.safetensors")
DownLoad(r"https://civitai.com/api/download/models/39164",str(user_home / r"stable-diffusion-webui" / r"extensions" / r"sd-webui-additional-networks" / r"models"/ r"lora"),r"backlighting.safetensors")
#strt webui

print("Done\nStarting Webui...")
os.chdir(user_home / r"stable-diffusion-webui")
while True:
  ret=subprocess.run([r"python3" ,r"launch.py",r"--precision",r"full",r"--no-half",r"--no-half-vae",r"--enable-insecure-extension-access",r"--medvram",r"--skip-torch-cuda-test",r"--enable-console-prompts",r"--ui-settings-file="+str(pathlib.Path(__file__).parent /r"config.json")])
  if(ret.returncode == 0 ): 
    del ret
    gc.collect()
  else :
    del ret

del os ,user_home ,pyexecutable ,subprocess