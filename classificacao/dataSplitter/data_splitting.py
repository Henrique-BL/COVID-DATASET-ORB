import os
import splitfolders

# caminho das imagens
input_path = 'classificacao/images_dataset/'
output_path = 'classificacao/images_split/'
##Renomeia todas as imagens da pasta para NOME_DA_PASTA+INDEX+EXTENSION
def rename_images(path):
    if os.path.exists(path):
        for dirpath , dirnames , filenames in os.walk(path):
            for index, file in enumerate(filenames):
                full_path = os.path.join(dirpath,file)
                extension = '.'+full_path.split('.')[-1] #-1 last element
                folder_name = os.path.basename(dirpath)
                newfilename = ''.join([folder_name+str(index), extension])
                os.rename(full_path,os.path.join(dirpath,newfilename))


if __name__ == '__main__':
    rename_images(input_path)      
    ##Split all images in test(20%) and train (80%)
    splitfolders.ratio(input=input_path, output=output_path,
    seed=2102, ratio=(.8, .2), group_prefix=None, move=False) # default values      