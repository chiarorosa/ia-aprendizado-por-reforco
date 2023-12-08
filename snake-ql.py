import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import count
import numpy as np
import os

# Configurações do jogo
LARGURA, ALTURA = 640, 480
TAMANHO_DA_SERPENTE = 20
VELOCIDADE = 5

# Configurações de aprendizado por reforço
GAMMA = 0.99
EPSILON_INICIAL = 0.9
EPSILON_FINAL = 0.05
EPSILON_DECAIMENTO = 200
REPLAY_BUFFER_CAPACIDADE = 10000
BATCH_SIZE = 32
num_episodes = 1000

# Função para verificar colisões
def is_collision(point1, point2, tamanho=TAMANHO_DA_SERPENTE):
    # Verifica se as coordenadas x e y de dois pontos estão dentro de 'tamanho' uma da outra
    return abs(point1[0] - point2[0]) < tamanho and abs(point1[1] - point2[1]) < tamanho

# Rede Neural
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Buffer de Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Função de seleção de ação
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            q_values = q_network(state)
            action = q_values.max(1)[1].item()
    else:
        action = random.randrange(4)
    return action

def get_state(snake, direction, food):
    head = snake[0]
    point_l = (head[0] - TAMANHO_DA_SERPENTE, head[1])
    point_r = (head[0] + TAMANHO_DA_SERPENTE, head[1])
    point_u = (head[0], head[1] - TAMANHO_DA_SERPENTE)
    point_d = (head[0], head[1] + TAMANHO_DA_SERPENTE)

    dir_l = direction == 'ESQUERDA'
    dir_r = direction == 'DIREITA'
    dir_u = direction == 'CIMA'
    dir_d = direction == 'BAIXO'

    state = [
        # Perigo direto à frente
        (dir_r and any(is_collision(point_r, part) for part in snake[1:])) or 
        (dir_l and any(is_collision(point_l, part) for part in snake[1:])) or 
        (dir_u and any(is_collision(point_u, part) for part in snake[1:])) or 
        (dir_d and any(is_collision(point_d, part) for part in snake[1:])),

        # Perigo direto à direita
        (dir_u and any(is_collision(point_r, part) for part in snake[1:])) or 
        (dir_d and any(is_collision(point_l, part) for part in snake[1:])) or 
        (dir_l and any(is_collision(point_u, part) for part in snake[1:])) or 
        (dir_r and any(is_collision(point_d, part) for part in snake[1:])),

        # Perigo direto à esquerda
        (dir_d and any(is_collision(point_r, part) for part in snake[1:])) or 
        (dir_u and any(is_collision(point_l, part) for part in snake[1:])) or 
        (dir_r and any(is_collision(point_u, part) for part in snake[1:])) or 
        (dir_l and any(is_collision(point_d, part) for part in snake[1:])),

        # Movimento direcional
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Posição da comida
        food[0] < head[0],  # Comida à esquerda
        food[0] > head[0],  # Comida à direita
        food[1] < head[1],  # Comida acima
        food[1] > head[1]   # Comida abaixo
    ]

    return np.array(state, dtype=int)

def take_action(snake, direction, action, last_direction, score, food):
    # Suponha que 'direction' seja a direção atual e 'last_direction' a última direção confirmada.
    
    # Determina a nova direção proposta com base na ação
    if action == 0:  # CIMA
        new_direction = 'CIMA'
    elif action == 1:  # BAIXO
        new_direction = 'BAIXO'
    elif action == 2:  # ESQUERDA
        new_direction = 'ESQUERDA'
    elif action == 3:  # DIREITA
        new_direction = 'DIREITA'

    # Verifica se a nova direção é oposta à última direção e impede a mudança se a serpente tiver mais de 1 segmento
    if len(snake) > 1:
        if (new_direction == 'CIMA' and last_direction == 'BAIXO') or \
           (new_direction == 'BAIXO' and last_direction == 'CIMA') or \
           (new_direction == 'ESQUERDA' and last_direction == 'DIREITA') or \
           (new_direction == 'DIREITA' and last_direction == 'ESQUERDA'):
            new_direction = last_direction  # impede a mudança para a direção oposta

    # Atualiza a direção atual com a nova direção se não for oposta
    if new_direction != last_direction:
        direction = new_direction

    # Atualiza a posição da cabeça da serpente com base na direção atualizada
    x, y = snake[0]
    if direction == 'CIMA':
        y -= TAMANHO_DA_SERPENTE
    elif direction == 'BAIXO':
        y += TAMANHO_DA_SERPENTE
    elif direction == 'ESQUERDA':
        x -= TAMANHO_DA_SERPENTE
    elif direction == 'DIREITA':
        x += TAMANHO_DA_SERPENTE

    # Verifica a colisão com as bordas
    done = x < 0 or x >= LARGURA or y < 0 or y >= ALTURA

    # Verifica colisão consigo mesma
    if (x, y) in snake[1:]:
        done = True

    # Verifica se a comida foi consumida
    eat = (x, y) == food
    if eat:
        score += 1
        # Gera nova posição para a comida
        food = (random.randint(0, (LARGURA - TAMANHO_DA_SERPENTE) // TAMANHO_DA_SERPENTE) * TAMANHO_DA_SERPENTE,
                random.randint(0, (ALTURA - TAMANHO_DA_SERPENTE) // TAMANHO_DA_SERPENTE) * TAMANHO_DA_SERPENTE)
    else:
        # Remove o último segmento do corpo se a comida não foi consumida
        snake.pop()

    # Adiciona a nova cabeça à serpente
    snake.insert(0, (x, y))

    # Define a recompensa
    reward = 0
    if eat:
        reward = 10
    elif done:
        reward = -10

    # Obtém o próximo estado
    next_state = get_state(snake, direction, food)

    return next_state, reward, done, score, food, direction

# Função de perda TD
def compute_td_loss(batch_size):
    if len(replay_buffer) < batch_size:
        return 0
    
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.uint8)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = q_network(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Inicialização do Pygame e preparação do ambiente
pygame.init()
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption('Snake Game')
clock = pygame.time.Clock()
fonte = pygame.font.SysFont('arial', 25)

# Inicialização do agente e do buffer de replay
q_network = QNetwork(11, 256, 4)
optimizer = optim.Adam(q_network.parameters())
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACIDADE)
epsilon = EPSILON_INICIAL

# Nome do arquivo onde o estado do modelo será salvo
MODEL_FILENAME = 'q_network.pth'
# Verifica se um modelo treinado já existe e carrega-o
if os.path.isfile(MODEL_FILENAME):
    q_network.load_state_dict(torch.load(MODEL_FILENAME))
    q_network.eval()  # coloca a rede em modo de avaliação
    # Se necessário, também carregue o valor de epsilon aqui
    # epsilon = ...

last_direction = None

def escolher_direcao_inicial():
    direcoes = ['CIMA', 'BAIXO', 'ESQUERDA', 'DIREITA']
    return random.choice(direcoes)

# Loop principal do jogo e treinamento
for episode in range(num_episodes):
    snake = [(LARGURA//2, ALTURA//2)]
     # Escolhe uma direção inicial aleatória para a serpente
    direction = escolher_direcao_inicial()  # 'CIMA', 'BAIXO', 'ESQUERDA' ou 'DIREITA' aleatoriamente
    last_direction = direction
    food = (random.randint(0, (LARGURA-TAMANHO_DA_SERPENTE)//TAMANHO_DA_SERPENTE) * TAMANHO_DA_SERPENTE,
            random.randint(0, (ALTURA-TAMANHO_DA_SERPENTE)//TAMANHO_DA_SERPENTE) * TAMANHO_DA_SERPENTE)
    score = 0
    state = get_state(snake, direction, food)
    for t in count():
        # Selecione e execute uma ação
        action = select_action(state, epsilon)
        next_state, reward, done, score, food, direction = take_action(snake, direction, action, last_direction, score, food)
        # Atualiza a última direção com a direção atual após o movimento
        last_direction = direction
        # Observe o novo estado
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Renderização e atualização da tela
        tela.fill((0, 0, 0))
        for part in snake:
            pygame.draw.rect(tela, pygame.Color('green'), pygame.Rect(part[0], part[1], TAMANHO_DA_SERPENTE, TAMANHO_DA_SERPENTE))
        pygame.draw.rect(tela, pygame.Color('red'), pygame.Rect(food[0], food[1], TAMANHO_DA_SERPENTE, TAMANHO_DA_SERPENTE))
        text = fonte.render("Score: {}".format(score), True, pygame.Color('white'))
        tela.blit(text, [0, 0])
        pygame.display.flip()
        clock.tick(VELOCIDADE)
        
        # Aprenda com a experiência (Replay Buffer)
        loss = compute_td_loss(BATCH_SIZE)

        # Salva o modelo após cada episódio ou após um número definido de episódios
        if episode % 100 == 0:  # Aqui estamos salvando a cada 100 episódios
            torch.save(q_network.state_dict(), MODEL_FILENAME)
            # Se necessário, também salve o valor de epsilon aqui
            # ...

        if done:
            break        

    # Atualize a política de exploração (epsilon)
    if epsilon > EPSILON_FINAL:
        epsilon -= (EPSILON_INICIAL - EPSILON_FINAL) / EPSILON_DECAIMENTO

    print(f"Episódio: {episode}, Score: {score}, Perda: {loss}")

    # salvar o modelo uma última vez após o treinamento ser concluído
    torch.save(q_network.state_dict(), MODEL_FILENAME)

pygame.quit()