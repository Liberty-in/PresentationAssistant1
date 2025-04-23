import keyboard
import streamlit as st

import cv2
import asyncio
import base64
import numpy as np
import json
import websockets
from PIL import Image
import io

create_page = st.Page("webst.py", title="Жесты", icon=":material/add_circle:")
delete_page = st.Page("pages/info.py", title="Статистика", icon=":material/delete:")

#pg = st.navigation([create_page, delete_page])


st.header("Распознавание жестов", divider="blue")
cl1,cl2,cl3 = st.columns(3)

c1 = cl1.container(border=True)
c2 = cl2.container(border=True)

bt = c1.button("Включить камеру")
c1.text("Включить камеру для отображения детекции")


bt1 = c2.button("Выключить камеру")
frame_placeholder = st.empty()

with st.expander("Нужна помощь?"):
    st.markdown('''
    В приложении можно: \n
        - распознавать жесты по данным с веб камеры \n
        - просматривать историю распознанных жестов \n
        - управлять презентацией с помощью жестов  \n
    Не забудьте разрешить сайту :blue[доступ к камере] :camera: \n
    Для переключения слайдов презентации выберите окно с презентацией как активное\n
    Чтобы посмотреть историю/статистику жестов, перейдите на вкладку :blue[Статистика]\n
    Доступные жесты для управления презентацией: \n
    - :v: - Предыдущий слайд 
    - :ok_hand: - Следующий слайд 
    - :+1: - Старт презентации 
    - :raised_hand_with_fingers_splayed: - Режим рисования 
    - :the_horns: - Стереть всё 
    - :hearts: - Первый слайд 
    - :open_hands: - Последний слайд 
    - :handshake: - Белый экран 
    - :open_hands: - Черный экран 
    
    ''')


def presentation_controls(number):
    actions = {
        0: ("Предыдущий слайд", lambda: keyboard.press_and_release('left')),
        1: ("Следующий слайд", lambda: keyboard.press_and_release('right')),
        2: ("Старт презентации", lambda: keyboard.press_and_release('f5')),
        3: ("Режим рисования", lambda: keyboard.press_and_release('ctrl+p')),
        4: ("Стереть всё", lambda: keyboard.press_and_release('e')),
        6: ("Первый слайд", lambda: keyboard.press_and_release('home')),
        5: ("Последний слайд", lambda: keyboard.press_and_release('end')),
        7: ("Белый экран", lambda: keyboard.press_and_release('w')),
        8: ("Черный экран", lambda: keyboard.press_and_release('b')),

    }

    action_name, action_func = actions[number]
    print(f"Выполняется действие: {action_name}")
    action_func()


async def send_frames():
    vid = cv2.VideoCapture(0)
    async with websockets.connect('ws://127.0.0.1:8000/ws') as websocket:
        while True:
            try:
                ret, frame = vid.read()
                if not ret:
                    break

                _, jpeg_frame = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg_frame.tobytes()
                await websocket.send(frame_bytes)
                response = await websocket.recv()
                #print(response)

                processed_image = base64.b64decode(response)
                processed_image = Image.open(io.BytesIO(processed_image))
                frame_placeholder.image(processed_image)


            except websockets.ConnectionClosedOK:
                print("Соединение было закрыто нормально")
                break
            except websockets.ConnectionClosedError as e:
                print(f"Ошибка соединения: {str(e)}")
                break
            except Exception as e:
                print(f"Произошла ошибка: {str(e)}")


if bt:
    asyncio.run(send_frames())

if bt1:
    frame_placeholder.empty()