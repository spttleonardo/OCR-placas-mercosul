import numpy as np
import cv2
from matplotlib import pyplot as plt
import visaoComputacional as visco


# Carregando imagem da placa
I1 = cv2.imread('./banco_de_imagens/nivel2/placa12.jpg')
cv2.imshow('Imagem de entrada', I1)

# Realizando resize para deixar todas as placas do mesmo tamanho
I1 = cv2.resize(I1, (500, 500))

# Realizando copia de I1 para homografia adiante
I_homo = I1.copy()

# Realizndo limiarizacao global
I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
L = 135
I2 = visco.limiarizacao_global_1( I2, L)

I3 = 255 - I2

# Aplicando o processo de fechamento na imagem
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
I4 = cv2.morphologyEx(I3, cv2.MORPH_CLOSE, kernel)

# Obtendo as bordas na imagem
I5 = cv2.Canny(I4, 50, 120, apertureSize=3)
#cv2.imshow('Imagem de borda após Canny', I5)

# obtendo regioes da imagem com as bordas definida
infoRegioes = visco.analisaRegioes5(I5)

# Trecho de codigo que filtra a area da placa como regiao de interesse
p1_r = 0
p2_r = 0
area2 = 0
for i in range(len(infoRegioes)):

    p1 = infoRegioes[i]['bb_point1']
    p2 = infoRegioes[i]['bb_point2']
    area = infoRegioes[i]['area']

    if 600 < area < 2000:
        if area2 < area: 
                area2 = area 
                p1_r, p2_r = p1, p2  # Armazenar os pontos para a região de interesse
        elif abs(area2 - area) < 40:
                area2 = area 
                p1_r, p2_r = p1, p2  # Armazenar os pontos para a região de interesse
        else:  
             pass

# Verificar se a região de interesse foi encontrada e se foi realizar um corte na imagem I1
if p1_r is not None and p2_r is not None:

    # Criar uma máscara com o mesmo tamanho da imagem, inicializada com zeros (preto)
    mask = np.zeros(I1.shape[:2], dtype=np.uint8)
    
    # Desenhar um retângulo branco (255) na máscara na região de interesse
    mask[p1_r[1]:p2_r[1], p1_r[0]:p2_r[0]] = 255

    # Aplicar a máscara na imagem original
    I_result = cv2.bitwise_and(I1, I1, mask=mask)
    
    # Exibir o resultado final
    cv2.imshow('Imagem com região de interesse isolada', I_result)

else:
    pass

# Alterando o espaço de cor da imagem com regiao de interesse que é a placa
I6 = cv2.cvtColor(I_result, cv2.COLOR_BGR2GRAY)

ret, I7 = cv2.threshold(I6, 165, 255,cv2.THRESH_BINARY)

# Obtendo os cotornos da regiao de interesse
contornos , hierarquia = cv2.findContours(I7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


# Criando matriz I8 que possui mesmas dimensoes da matriz com regiao de interesse
I8 = np.zeros(I7.shape, np.uint8)

# filtrando a placa através do contorno difinido pelo autor
for i in range(len(contornos)):

    if len(contornos[i][:,0,0]) > 600:
        cv2.drawContours(I8, contornos, i, [255,255,255])

cv2.imshow('Imagem com contorno selecionador', I8)

# Obtendo novamente o contorno da regiao da placa para realizar o algoritimo 
# de Ramer-Douglas-Peucker
contornos , hierarquia = cv2.findContours(I8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contornos[0]

# algoritimo de Ramer-Douglas-Peucker que pega todos os pontos de contorno e aproxima para
# 4 pontos 
for k in np.linspace(0.01, 0.1, 100):
    epsilon = k*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        break

cv2.polylines(I1, [np.int32(approx)], True, (128,255,255), 3, cv2.LINE_AA)
cv2.imshow('Imagem result', I1)

# -------------------- homografia ----------------------
#Definindo tamanho final da imagem
n_linhas, n_colunas = 300, 1550

# Transformar approx em uma lista de pontos, ou seja, removendo uma dimensao
points = [pt[0] for pt in approx]  

# Separação em listas de acordo com coordenadas x para criar os grupos
lista_pixel_menor, lista_pixel_maior = [], []

# Separar os pontos em duas listas com base no valor de x
for item in points:
    if item[0] < 250:  # Critério para separação baseado na coordenada x
        lista_pixel_menor.append((item[0], item[1])) 
    else:
        lista_pixel_maior.append((item[0], item[1])) 


# Ordenar as listas primeiro por coordenada y, depois por coordenada x
lista_pixel_menor_sorted = sorted(lista_pixel_menor, key=lambda point: (point[1], point[0]))  
lista_pixel_maior_sorted = sorted(lista_pixel_maior, key=lambda point: (point[1], point[0])) 


# Verifica se há elementos suficientes para definir pontos de homografia
if len(lista_pixel_menor) >= 2 and len(lista_pixel_maior) >= 2:

    # Definir os pontos para homografia com base nas listas organizadas
    pts_org = np.array([
        lista_pixel_menor_sorted[0], 
        lista_pixel_menor_sorted[1],  
        lista_pixel_maior_sorted[1],
        lista_pixel_maior_sorted[0] 
    ], dtype=np.float32)

    # Pontos de destino da homografia
    pts_dst = np.array([[0, 0], [0, n_linhas - 1], [n_colunas - 1, n_linhas - 1], [n_colunas - 1, 0]], dtype=np.float32)

    # Calcular e aplicar a matriz de homografia
    H = visco.homografia(pts_org, pts_dst)
    I_posHomo = cv2.warpPerspective(I_homo, H, (n_colunas, n_linhas))
    #cv2.imshow('Resultado da Homografia', I_posHomo)
else:
    pass

#--------------- Encontrando regioes das letras na placa -------------------

# Alterando espaço de cor
I1_placa = cv2.cvtColor(I_posHomo, cv2.COLOR_BGR2GRAY)
_, I2_placa = cv2.threshold(I1_placa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

I2_placa = 255-I2_placa

#Realizando remoção dos objetos nas bordas através
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,8))
I3_placa = cv2.dilate(I2_placa, kernel)

Mask = I3_placa.copy()
Marker = Mask.copy()
Marker[5:-20, 5:-20] = 0

# # chamando função para realizar a operação de dilatação e bitwise_an 
I_reco = visco.imreconstruction(Mask, Marker)

I4_placa = I3_placa - I_reco
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
I5_placa = cv2.erode(I4_placa, kernel)

# Realizando processo de aumento de nitidez
# Suavizando imagem com o filtro gaussiano
w = 9
sigma = (w-1)/6
Is = cv2.GaussianBlur(I5_placa, (w,w), sigma)

#Obtendo mascara através da subtracao da imagem suavizada e imagem da placa
Ig = I5_placa - Is

# adicionando a mascara à imagem original para realizar o agucamento da imagem
k = 1.5
Ifinal_placa = I5_placa + k*Ig


Ifinal_placa = Ifinal_placa.astype(np.uint8)
cv2.imshow('Imagem com o aumento de nitidez', Ifinal_placa)

# obtendo as regiões da imagem final das placas
infoRegioes2 = visco.analisaRegioes5(Ifinal_placa)


# ----------------------- criando templates  -----------------------
# Carregando imagem dos templates
template = cv2.imread('./banco_de_imagens/fonte_mercosul.png')

# Convertendo de BGR para escala de cinza
temp1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#Realizando limiar de otsu e obtendo o negativo
_, temp2 = cv2.threshold(temp1, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU )
temp3 = 255 - temp2
cv2.imshow('templates', temp3)

#Obtendo os componentes coenctados e organizando os mesmos de acordo com coordenada x
infoRegioes  = visco.analisaRegioes(temp3)

# definindo uma tolerancia para realizar a ordenacao em y
tolerancia = 10

# Realizando sorted em y no inforegioes
infoRegioes = sorted(infoRegioes, key=lambda regiao: (regiao['centroide'][1]))

# Organizando as regiões por linha
linhas = []
linha_atual = []

for i, regiao in enumerate(infoRegioes):
    if i == 0:
         linha_atual.append(regiao)
    else:
        if abs(regiao['centroide'][1] - linha_atual[-1]['centroide'][1] <= tolerancia):
            linha_atual.append(regiao)
        else:
            linha_atual = sorted(linha_atual, key=lambda r: r['centroide'][0])
            linhas.append(linha_atual)
            linha_atual = [regiao]

if linha_atual:
    linha_atual = sorted(linha_atual, key=lambda r: r['centroide'][0])
    linhas.append(linha_atual)


# Concatenando todas as linhas para obter a lista final em ordem
infoRegioes_ordenadas = [regiao for linha in linhas for regiao in linha]

# definindo o numero de pontos para realizar o processo de normalizacao nas curvas
numero_de_pontos = 600

# dicionario para armazenar as letras contidas na lista e suas curvas de distancia 
dic = {}
lista = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
    'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Laco de repeticao para armazenar a letra e a curva no dicionario
for i, letra in enumerate(lista):
    curva_distancia2 = infoRegioes_ordenadas[i]['curva_distancia']
    dic[letra] = curva_distancia2 


# lista para armazenar as letras e seus centroides na posicao x
letras_seq = []

# Loop para cada curva em infoRegioes2, ou seja, da placa
for j in range(len(infoRegioes2)):
    lista = []
    lista_name = []
    area = infoRegioes2[j]['area']
    
    # condicional para filtra elementos na imagem que não são letras
    if area > 7000:

        # Realizando configuracoes dna curva de distancia 1 para ficar igual a curva de distancia 2
        curva_distancia1 = infoRegioes2[j]['curva_distancia']
        N1 = len(curva_distancia1)
        curva_distancia1_interpolada = np.interp(np.linspace(0, N1 - 1, numero_de_pontos), np.arange(0, N1), curva_distancia1)
        y1 = curva_distancia1_interpolada - np.mean(curva_distancia1_interpolada)
        y1n = y1 / np.sqrt(np.sum(y1 ** 2))

        # Loop para calcular a similaridade com cada curva de referência
        for name, curva in dic.items():

            # Realizando configuracoes dna curva de distancia 2 para ficar igual a curva de distancia 1
            N2 = len(curva)
            curva_distancia2_interpolada = np.interp(np.linspace(0, N2 - 1, numero_de_pontos), np.arange(0, N2), curva)
            y2 = curva_distancia2_interpolada - np.mean(curva_distancia2_interpolada)
            y2n = y2 / np.sqrt(np.sum(y2 ** 2))

            # Cálculo da correlação cruzada
            correlacao = np.zeros(len(y1n))
            for k in range(len(y1n)):
                correlacao[k] = np.sum(y1n * np.roll(y2n, k))

            # Encontra a similaridade máxima para a curva em questão
            similaridade = np.max(correlacao)
            lista.append(similaridade)
            lista_name.append(name)

        # armazeando a letra que foi correspondida com mais valor de similaridade, junto com seu centroide
        indice = np.argmax(lista)
        letras_seq.append((lista_name[indice], infoRegioes2[j]['centroide'][0]))


# Ordenando letraas_seq pela posição x
letras_seq_ordenada = sorted(letras_seq, key=lambda item: item[1])

# Concatena as letras em uma string, mantendo duplicatas
resultado = ''.join(letra for letra, _ in letras_seq_ordenada)

print(f'A placa do veiculo e: {resultado}')

cv2.waitKey(0)