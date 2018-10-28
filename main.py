"""
An implementation of the game interface

@author: Blaise Wang
"""

import copy
import pickle
import threading

import numpy as np
import wx

from game import Board

# from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet

N = 8
WIN_WIDTH = 800
WIN_HEIGHT = 450
HEIGHT_OFFSET = 50
BANNER_WIDTH = 300
BANNER_HEIGHT = 100
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 32
ROW_LIST_MARGIN = -40
COLUMN_LIST_MARGIN = 25
BUTTON_WIDTH_MARGIN = 6
BUTTON_HEIGHT_MARGIN = 45


class OthelloFrame(wx.Frame):
    current_move = 0
    has_set_ai_player = False
    is_banner_displayed = False
    is_analysis_displayed = False

    block_length = int((WIN_HEIGHT - 90) / N)
    piece_radius = (block_length >> 1) - 3
    inner_circle_radius = piece_radius - 4
    half_button_width = (BUTTON_WIDTH - BUTTON_WIDTH_MARGIN) >> 1

    mcts_player = None

    line_list = []
    row_list = []
    column_list = []
    chess_record = []
    row_name_list = ['15', '14', '13', '12', '11', '10', ' 9', ' 8', ' 7', ' 6', ' 5', ' 4', ' 3', ' 2', ' 1']
    column_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    def __init__(self):
        self.n = N
        self.board = Board(self.n)
        self.thread = threading.Thread()
        self.row_name_list = self.row_name_list[15 - self.n: 15]
        self.column_name_list = self.column_name_list[0: self.n]
        self.grid_length = self.block_length * (self.n - 1)
        self.grid_position_x = ((WIN_WIDTH - self.grid_length) >> 1) + 15
        self.grid_position_y = (WIN_HEIGHT - self.grid_length - HEIGHT_OFFSET) >> 1
        self.button_position_x = (self.grid_position_x + ROW_LIST_MARGIN - BUTTON_WIDTH) >> 1
        self.second_button_position_x = self.button_position_x + self.half_button_width + BUTTON_WIDTH_MARGIN

        for i in range(0, self.grid_length + 1, self.block_length):
            self.line_list.append((i + self.grid_position_x, self.grid_position_y, i + self.grid_position_x,
                                   self.grid_position_y + self.grid_length - 1))
            self.line_list.append((self.grid_position_x, i + self.grid_position_y,
                                   self.grid_position_x + self.grid_length - 1, i + self.grid_position_y))
            self.row_list.append((self.grid_position_x + ROW_LIST_MARGIN, i + self.grid_position_y - 8))
            self.column_list.append(
                (i + self.grid_position_x, self.grid_position_y + self.grid_length + COLUMN_LIST_MARGIN))

        wx.Frame.__init__(self, None, title="Non-Internet Reversi",
                          pos=((wx.DisplaySize()[0] - WIN_WIDTH) >> 1, (wx.DisplaySize()[1] - WIN_HEIGHT) / 2.5),
                          size=(WIN_WIDTH, WIN_HEIGHT), style=wx.CLOSE_BOX)
        button_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        image_font = wx.Font(25, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)

        self.replay_button = wx.Button(self, label="Replay",
                                       pos=(self.button_position_x, self.grid_position_y + BUTTON_HEIGHT_MARGIN),
                                       size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.black_button = wx.Button(self, label="●",
                                      pos=(self.button_position_x, self.grid_position_y + 2 * BUTTON_HEIGHT_MARGIN),
                                      size=(self.half_button_width, BUTTON_HEIGHT))
        self.white_button = wx.Button(self, label="○",
                                      pos=(
                                          self.second_button_position_x,
                                          self.grid_position_y + 2 * BUTTON_HEIGHT_MARGIN),
                                      size=(self.half_button_width, BUTTON_HEIGHT))
        self.ai_hint_button = wx.Button(self, label="Hint",
                                        pos=(self.button_position_x, self.grid_position_y + 3 * BUTTON_HEIGHT_MARGIN),
                                        size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.analysis_button = wx.Button(self, label="Analysis",
                                         pos=(self.button_position_x, self.grid_position_y + 4 * BUTTON_HEIGHT_MARGIN),
                                         size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.black_text = wx.StaticText(self, label="●", pos=(
            self.button_position_x + 35, self.grid_position_y + 5 * BUTTON_HEIGHT_MARGIN + 10), size=wx.Size(100, 20))
        self.black_number = wx.StaticText(self, label="", pos=(
            self.button_position_x + 100, self.grid_position_y + 5 * BUTTON_HEIGHT_MARGIN + 10), size=wx.Size(100, 20))
        self.white_text = wx.StaticText(self, label="○", pos=(
            self.button_position_x + 35, self.grid_position_y + 6 * BUTTON_HEIGHT_MARGIN + 10), size=wx.Size(100, 20))
        self.white_number = wx.StaticText(self, label="", pos=(
            self.button_position_x + 100, self.grid_position_y + 6 * BUTTON_HEIGHT_MARGIN + 10), size=wx.Size(100, 20))
        self.replay_button.SetFont(button_font)
        self.ai_hint_button.SetFont(button_font)
        self.analysis_button.SetFont(button_font)
        self.black_button.SetFont(image_font)
        self.white_button.SetFont(image_font)
        self.replay_button.Disable()
        try:
            policy_param = pickle.load(open('best.model', 'rb'), encoding='bytes')
            # self.mcts_player = MCTSPlayer(PolicyValueNet(self.n, net_params=policy_param).policy_value_func, c_puct=5,
            #                               n_play_out=400)
            self.black_button.Enable()
            self.white_button.Enable()
            self.ai_hint_button.Enable()
            self.analysis_button.Enable()
        except IOError as _:
            self.black_button.Disable()
            self.white_button.Disable()
            self.ai_hint_button.Disable()
            self.analysis_button.Disable()
        self.initialize_user_interface()

    def on_replay_button_click(self, _):
        if not self.thread.is_alive():
            self.board.initialize()
            self.current_move = 0
            self.has_set_ai_player = False
            self.chess_record.clear()
            self.draw_board()
            self.draw_chess()
            self.replay_button.Disable()
            if self.mcts_player is not None:
                self.black_button.Enable()
                self.white_button.Enable()
                self.ai_hint_button.Enable()
                self.analysis_button.Enable()

    def on_black_button_click(self, _):
        self.black_button.Disable()
        self.white_button.Disable()
        self.has_set_ai_player = True

    def on_white_button_click(self, _):
        self.black_button.Disable()
        self.white_button.Disable()
        self.has_set_ai_player = True
        self.thread = threading.Thread(target=self.ai_next_move, args=())
        self.thread.start()

    def on_ai_hint_button_click(self, _):
        if not self.thread.is_alive():
            self.black_button.Disable()
            self.white_button.Disable()
            self.ai_next_move()

    def on_analysis_button_click(self, _):
        if not self.thread.is_alive():
            moves, probability = copy.deepcopy(self.mcts_player).get_action(self.board, return_probability=1)
            move_list = [(moves[i], p) for i, p in enumerate(probability) if p > 0]
            if len(move_list) > 0:
                self.draw_possible_moves(move_list)
                self.is_analysis_displayed = True
                self.analysis_button.Disable()

    def on_paint(self, _):
        dc = wx.PaintDC(self)
        dc.SetBackground(wx.Brush(wx.WHITE_BRUSH))
        dc.Clear()
        self.draw_board()
        self.draw_chess()
        self.update_number()

    def ai_next_move(self):
        move, move_probabilities = self.mcts_player.get_action(self.board)
        x, y = self.board.move_to_location(move)
        self.board.add_move(x, y)
        if self.is_analysis_displayed:
            self.repaint_board()
        self.analysis_button.Enable()
        self.draw_move(y, x)

    def disable_buttons(self):
        if self.board.has_winner() != -1:
            self.ai_hint_button.Disable()
            self.analysis_button.Disable()

    def initialize_user_interface(self):
        self.board = Board(self.n)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_UP, self.on_click)
        self.Bind(wx.EVT_BUTTON, self.on_replay_button_click, self.replay_button)
        self.Bind(wx.EVT_BUTTON, self.on_black_button_click, self.black_button)
        self.Bind(wx.EVT_BUTTON, self.on_white_button_click, self.white_button)
        self.Bind(wx.EVT_BUTTON, self.on_ai_hint_button_click, self.ai_hint_button)
        self.Bind(wx.EVT_BUTTON, self.on_analysis_button_click, self.analysis_button)
        self.Centre()
        self.Show(True)

    def repaint_board(self):
        self.draw_board()
        self.draw_chess()
        self.is_banner_displayed = False
        self.is_analysis_displayed = False

    def draw_board(self):
        dc = wx.ClientDC(self)
        dc.SetPen(wx.Pen(wx.WHITE))
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(self.grid_position_x - self.block_length, self.grid_position_y - self.block_length,
                         self.grid_length + self.block_length * 2, self.grid_length + self.block_length * 2)
        dc.SetPen(wx.Pen(wx.BLACK, width=2))
        dc.DrawLineList(self.line_list)
        dc.SetFont(wx.Font(13, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False))
        dc.DrawTextList(self.row_name_list, self.row_list)
        dc.DrawTextList(self.column_name_list, self.column_list)
        dc.SetBrush(wx.Brush(wx.BLACK))
        if self.n % 2 == 1:
            dc.DrawCircle(self.grid_position_x + self.block_length * (self.n >> 1),
                          self.grid_position_y + self.block_length * (self.n >> 1), 4)
        if self.n == 15:
            dc.DrawCircle(self.grid_position_x + self.block_length * 3, self.grid_position_y + self.block_length * 3, 4)
            dc.DrawCircle(self.grid_position_x + self.block_length * 3, self.grid_position_y + self.block_length * 11,
                          4)
            dc.DrawCircle(self.grid_position_x + self.block_length * 11, self.grid_position_y + self.block_length * 3,
                          4)
            dc.DrawCircle(self.grid_position_x + self.block_length * 11, self.grid_position_y + self.block_length * 11,
                          4)

    def draw_possible_moves(self, possible_move):
        dc = wx.ClientDC(self)
        for move, p in possible_move:
            y, x = self.board.move_to_location(move)
            dc.SetBrush(wx.Brush(wx.Colour(28, 164, 252, alpha=14 if int(p * 230) < 14 else int(p * 230))))
            dc.SetPen(wx.Pen(wx.Colour(28, 164, 252, alpha=230)))
            dc.DrawCircle(self.grid_position_x + x * self.block_length, self.grid_position_y + y * self.block_length,
                          self.piece_radius)

    def draw_chess(self):
        dc = wx.ClientDC(self)
        self.disable_buttons()
        for x, y in np.ndindex(self.board.chess[0:self.n, 0:self.n].shape):
            if self.board.chess[y, x] > 0:
                dc.SetBrush(wx.Brush(wx.BLACK if self.board.chess[y, x] == 1 else wx.WHITE))
                dc.DrawCircle(self.grid_position_x + x * self.block_length,
                              self.grid_position_y + y * self.block_length, self.piece_radius)
        if self.current_move > 0:
            x, y = self.chess_record[self.current_move - 1]
            dc.SetBrush(wx.Brush(wx.BLACK if self.board.chess[y, x] == 1 else wx.WHITE))
            dc.SetPen(wx.Pen(wx.WHITE if self.board.chess[y, x] == 1 else wx.BLACK))
            x = self.grid_position_x + x * self.block_length
            y = self.grid_position_y + y * self.block_length
            dc.DrawCircle(x, y, self.inner_circle_radius)

    def draw_move(self, x: int, y: int) -> bool:
        self.current_move += 1
        self.chess_record.append((x, y))
        self.draw_chess()
        winner = self.board.has_winner()
        if winner != -1:
            self.disable_buttons()
            self.draw_banner(winner)
            return True
        return False

    def draw_banner(self, result: int):
        w = 216
        if result == 1:
            string = "BLACK WIN"
        elif result == 2:
            string = "WHITE WIN"
        else:
            string = "DRAW"
            w = 97
        x = (self.grid_position_x + ((self.grid_length - w) >> 1))
        dc = wx.ClientDC(self)
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(self.grid_position_x + ((self.grid_length - BANNER_WIDTH) >> 1),
                         self.grid_position_y + ((self.grid_length - BANNER_HEIGHT) >> 1), BANNER_WIDTH, BANNER_HEIGHT)
        dc.SetPen(wx.Pen(wx.BLACK))
        dc.SetFont(wx.Font(40, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False))
        dc.DrawText(string, x, (self.grid_position_y + ((self.grid_length - 40) >> 1)))
        self.is_banner_displayed = True

    def update_number(self):
        black, white = self.board.get_color_number()
        self.black_number.SetLabel(str(black))
        self.white_number.SetLabel(str(white))

    def on_click(self, e):
        if not self.thread.is_alive():
            if self.board.winner == -1:
                if self.is_analysis_displayed:
                    self.repaint_board()
                x, y = e.GetPosition()
                x = x - self.grid_position_x + (self.block_length >> 1)
                y = y - self.grid_position_y + (self.block_length >> 1)
                if x > 0 and y > 0:
                    x = int(x / self.block_length)
                    y = int(y / self.block_length)
                    if 0 <= x < self.n and 0 <= y < self.n:
                        if self.board.chess[y, x] == 0:
                            if self.board.location_to_move(y, x) in self.board.get_available_moves(
                                    self.board.get_current_player()):
                                if self.mcts_player is not None:
                                    self.analysis_button.Enable()
                                    self.black_button.Disable()
                                    self.white_button.Disable()
                                self.board.add_move(y, x)
                                has_end = self.draw_move(x, y)
                                self.replay_button.Enable()
                                self.update_number()
                                if self.has_set_ai_player and not has_end:
                                    self.thread = threading.Thread(target=self.ai_next_move, args=())
                                    self.thread.start()
            elif self.is_banner_displayed:
                self.repaint_board()


if __name__ == '__main__':
    app = wx.App(False)
    OthelloFrame()
    app.MainLoop()
