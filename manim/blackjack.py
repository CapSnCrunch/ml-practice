from manim import *

class Card(VGroup):
    SUIT_SYMBOLS = {
        'S': '♠',
        'H': '♥',
        'D': '♦',
        'C': '♣',
    }
    SUIT_COLORS = {
        'S': WHITE,
        'C': WHITE,
        'H': RED,
        'D': RED,
    }

    def __init__(self, value, suit, color=None, **kwargs):
        super().__init__(**kwargs)
        card_rect = RoundedRectangle(corner_radius=0.05, width=1.2, height=1.7, stroke_width=2, fill_opacity=1, fill_color=WHITE)
        suit_symbol = self.SUIT_SYMBOLS.get(suit.upper(), '?')
        card_color = color if color else self.SUIT_COLORS.get(suit.upper(), WHITE)
        value_text = Text(str(value), font_size=36, color=card_color)
        # Top-left
        top_left = value_text.copy().move_to(card_rect.get_corner(UL) + DOWN*0.20 + RIGHT*0.22)
        # Bottom-right (move further inside)
        bottom_right = value_text.copy().move_to(card_rect.get_corner(DR) + UP*0.20 + LEFT*0.22)
        # Main suit in center (larger)
        center_suit = Text(suit_symbol, font_size=60, color=card_color).move_to(card_rect.get_center())
        self.add(card_rect, top_left, bottom_right, center_suit)

class AnimateSingleCard(Scene):
    def construct(self):
        card = Card("A", "H")  # Ace of Hearts
        card.move_to(LEFT * 4)
        self.play(FadeIn(card))
        self.wait(0.5)
        self.play(card.animate.move_to(RIGHT * 4), run_time=2)
        self.wait(0.5)
        self.play(FadeOut(card))
