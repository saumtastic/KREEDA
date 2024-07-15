import streamlit as st
from ppa import HandballGame

st.title("Handball Game")
st.write("Play a fun handball game with your friends or against the computer!")

game_mode = st.selectbox("Select Game Mode", ["Two Players (Local)", "Play with Computer", "Create Private Room"])

player1_name = st.text_input("Player 1 Name", "Player 1")
player2_name = st.text_input("Player 2 Name", "Player 2" if game_mode != "Play with Computer" else "Computer")

if st.button("Start Game"):
    game = HandballGame()
    if game_mode == "Two Players (Local)":
        score = game.play_game(mode="local", player1=player1_name, player2=player2_name)
    elif game_mode == "Play with Computer":
        score = game.play_game(mode="computer", player1=player1_name, player2="Computer")
    else:
        # Implement private room functionality
        pass

    st.write(f"Game Over! Final Score:\n{player1_name}: {score[0]}\n{player2_name}: {score[1]}")
    if score[0] > score[1]:
        st.write(f"Winner: {player1_name}")
    elif score[1] > score[0]:
        st.write(f"Winner: {player2_name}")
    else:
        st.write("It's a tie!")
