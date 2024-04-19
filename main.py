from game import Game

def main(auto_start_game:bool = False):
    try:
        game = Game(1920, 1080)
        if auto_start_game:
            game.set_to_running_state()
        game.run()
    except Exception as e:
        print(e)
        main(True)
    
if __name__ == '__main__':
    main()
    