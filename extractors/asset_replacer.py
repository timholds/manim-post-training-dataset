"""
Smart asset replacer for Manim training data.
Replaces binary assets with appropriate Manim primitives based on context.
"""

import ast
import re
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AssetReplacer(ast.NodeTransformer):
    """
    Replaces ImageMobject and SVGMobject calls with context-appropriate Manim primitives.
    """
    
    def __init__(self):
        self.replacements = []
        self.failed_replacements = []
        
        # Asset name patterns to replacement strategies
        self.replacement_strategies = {
            # Characters/People
            r'(farmer|rabbit|turtle|bee|kid|student|professor|person|character)': 'character',
            r'(kantorovich|dantzig|euler|gauss)\.jpg': 'portrait',
            
            # Icons/Symbols
            r'(factory|house|building|store)\.svg': 'building',
            r'(pause|play|stop|button)\.svg': 'ui_button',
            r'(eq|leq|geq|iff|implies)\.svg': 'math_symbol',
            r'(python|java|cpp|haskell|logo)\.svg': 'logo',
            
            # Objects
            r'(carrot|apple|food|item|product|potato|fertilizer)': 'item',
            r'(car|truck|vehicle)': 'vehicle',
            
            # Backgrounds/Images
            r'(background|bg|scene)\.png': 'background',
            r'(black|white|green|blue)\.png': 'solid_color',
            r'(farm|field|landscape).*\.png': 'background',
            r'(proof|paper|document|handbook|performance).*\.png': 'document',
            
            # Animals
            r'(dragonfly|giraffe|mud|soap).*\.png': 'generic_shape',
            
            # Diagrams
            r'(diagram|graph|chart|simplex|duality)': 'diagram',
            r'(cache|cpu|memory|hardware)': 'technical',
            r'(thingy|arrow|pointer)\.png': 'arrow',
            
            # Misc xiaoxiae patterns
            r'assets/.*\.png': 'generic_shape',
            r'assets/.*\.svg': 'generic_shape',
            r'\d+.*\.png': 'generic_shape',  # numbered images like 2.png
            
            # Default
            r'thumbnail': 'thumbnail',
        }
    
    def visit_Call(self, node):
        """Visit Call nodes to find ImageMobject and SVGMobject."""
        if self._is_asset_call(node):
            filename = self._extract_filename(node)
            if filename:
                replacement = self._create_replacement(filename, node)
                if replacement:
                    self.replacements.append(filename)
                    return replacement
                else:
                    self.failed_replacements.append(filename)
        
        self.generic_visit(node)
        return node
    
    def _is_asset_call(self, node):
        """Check if node is ImageMobject, SVGMobject, or OpenGLImageMobject call."""
        return (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id in ['ImageMobject', 'SVGMobject', 'OpenGLImageMobject'])
    
    def _extract_filename(self, node):
        """Extract filename from asset call."""
        if node.args and len(node.args) > 0:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                return arg.value
            elif isinstance(arg, ast.Str):  # Python 3.7
                return arg.s
        return None
    
    def _determine_strategy(self, filename: str) -> str:
        """Determine replacement strategy based on filename."""
        filename_lower = filename.lower()
        
        for pattern, strategy in self.replacement_strategies.items():
            if re.search(pattern, filename_lower):
                return strategy
        
        # Default based on extension
        if filename.endswith('.svg'):
            return 'generic_icon'
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            return 'generic_image'
        
        return 'generic_shape'
    
    def _create_replacement(self, filename: str, original_node: ast.Call) -> Optional[ast.Call]:
        """Create appropriate Manim primitive replacement."""
        strategy = self._determine_strategy(filename)
        
        # Extract modifiers from original (like .set_height(), .scale(), etc)
        modifiers = self._extract_modifiers(original_node)
        
        # Create base replacement based on strategy
        if strategy == 'character':
            replacement = self._create_character(filename)
        elif strategy == 'portrait':
            replacement = self._create_portrait(filename)
        elif strategy == 'building':
            replacement = self._create_building(filename)
        elif strategy == 'ui_button':
            replacement = self._create_ui_button(filename)
        elif strategy == 'math_symbol':
            replacement = self._create_math_symbol(filename)
        elif strategy == 'logo':
            replacement = self._create_logo(filename)
        elif strategy == 'item':
            replacement = self._create_item(filename)
        elif strategy == 'vehicle':
            replacement = self._create_vehicle(filename)
        elif strategy == 'background':
            replacement = self._create_background(filename)
        elif strategy == 'solid_color':
            replacement = self._create_solid_color(filename)
        elif strategy == 'diagram':
            replacement = self._create_diagram(filename)
        elif strategy == 'technical':
            replacement = self._create_technical_diagram(filename)
        elif strategy == 'thumbnail':
            replacement = self._create_thumbnail(filename)
        elif strategy == 'document':
            replacement = self._create_document(filename)
        elif strategy == 'arrow':
            replacement = self._create_arrow(filename)
        else:
            replacement = self._create_generic_shape(filename)
        
        # Apply original modifiers
        return self._apply_modifiers(replacement, modifiers)
    
    def _extract_modifiers(self, node: ast.Call) -> list:
        """Extract method calls like .set_height() from the original."""
        # This is complex in AST form, so we'll handle basic cases
        # In practice, we'd need to trace through the parent nodes
        return []
    
    def _create_character(self, filename: str) -> ast.Call:
        """Create a simple character representation."""
        # VGroup with circle (head) and rectangle (body)
        return ast.parse("""
VGroup(
    Circle(radius=0.3, color=BLUE),
    Rectangle(width=0.4, height=0.6, color=BLUE).shift(DOWN * 0.6)
)
""").body[0].value
    
    def _create_portrait(self, filename: str) -> ast.Call:
        """Create a portrait placeholder with name."""
        name = filename.split('/')[-1].split('.')[0].title()
        return ast.parse(f"""
VGroup(
    Rectangle(width=2, height=3, color=GREY),
    Text("{name}", font_size=24).shift(DOWN * 2)
)
""").body[0].value
    
    def _create_building(self, filename: str) -> ast.Call:
        """Create a simple building shape."""
        if 'factory' in filename.lower():
            # Factory with chimney
            return ast.parse("""
VGroup(
    Rectangle(width=2, height=1.5, color=GREY),
    Rectangle(width=0.3, height=0.5, color=DARK_GREY).shift(UP * 1.25 + RIGHT * 0.7)
)
""").body[0].value
        else:
            # Generic building
            return ast.parse("""
VGroup(
    Rectangle(width=1.5, height=2, color=GREY),
    Triangle(color=RED).scale(0.8).shift(UP * 1.3)
)
""").body[0].value
    
    def _create_ui_button(self, filename: str) -> ast.Call:
        """Create UI button shape."""
        if 'pause' in filename.lower():
            return ast.parse("""
VGroup(
    Rectangle(width=0.1, height=0.3, color=WHITE).shift(LEFT * 0.1),
    Rectangle(width=0.1, height=0.3, color=WHITE).shift(RIGHT * 0.1)
)
""").body[0].value
        elif 'play' in filename.lower():
            return ast.parse("Triangle(color=WHITE).rotate(-PI/2).scale(0.3)").body[0].value
        else:
            return ast.parse("Circle(radius=0.3, color=WHITE)").body[0].value
    
    def _create_math_symbol(self, filename: str) -> ast.Call:
        """Create mathematical symbol using MathTex."""
        symbols = {
            'eq': '=',
            'leq': r'\leq',
            'geq': r'\geq',
            'iff': r'\iff',
            'implies': r'\implies'
        }
        
        for key, symbol in symbols.items():
            if key in filename.lower():
                return ast.parse(f'MathTex(r"{symbol}")').body[0].value
        
        return ast.parse('MathTex("?")').body[0].value
    
    def _create_logo(self, filename: str) -> ast.Call:
        """Create programming language logo placeholder."""
        name = filename.split('/')[-1].split('.')[0].upper()
        return ast.parse(f"""
VGroup(
    RoundedRectangle(corner_radius=0.2, width=1.5, height=1.5, color=BLUE),
    Text("{name[:2]}", font_size=36, color=WHITE)
)
""").body[0].value
    
    def _create_item(self, filename: str) -> ast.Call:
        """Create simple item representation."""
        if 'carrot' in filename.lower():
            # Orange triangle for carrot
            return ast.parse("Triangle(color=ORANGE).scale(0.5)").body[0].value
        else:
            # Generic item
            return ast.parse("Circle(radius=0.4, color=GREEN)").body[0].value
    
    def _create_vehicle(self, filename: str) -> ast.Call:
        """Create simple vehicle shape."""
        return ast.parse("""
VGroup(
    Rectangle(width=2, height=0.8, color=RED),
    Circle(radius=0.3, color=BLACK).shift(DOWN * 0.6 + LEFT * 0.6),
    Circle(radius=0.3, color=BLACK).shift(DOWN * 0.6 + RIGHT * 0.6)
)
""").body[0].value
    
    def _create_background(self, filename: str) -> ast.Call:
        """Create background rectangle."""
        return ast.parse("""
Rectangle(
    width=config.frame_width,
    height=config.frame_height,
    color=DARK_GREY,
    fill_opacity=0.5
)
""").body[0].value
    
    def _create_solid_color(self, filename: str) -> ast.Call:
        """Create solid color rectangle based on filename."""
        colors = {
            'black': 'BLACK',
            'white': 'WHITE',
            'green': 'GREEN',
            'blue': 'BLUE',
            'red': 'RED'
        }
        
        color = 'GREY'  # default
        for name, manim_color in colors.items():
            if name in filename.lower():
                color = manim_color
                break
        
        return ast.parse(f"""
Rectangle(
    width=config.frame_width,
    height=config.frame_height,
    color={color},
    fill_opacity=1
)
""").body[0].value
    
    def _create_diagram(self, filename: str) -> ast.Call:
        """Create placeholder diagram."""
        return ast.parse("""
VGroup(
    Rectangle(width=4, height=3, color=WHITE),
    Text("Diagram", font_size=24, color=GREY)
)
""").body[0].value
    
    def _create_technical_diagram(self, filename: str) -> ast.Call:
        """Create technical diagram placeholder."""
        return ast.parse("""
VGroup(
    *[Rectangle(width=0.8, height=0.3, color=BLUE).shift(DOWN * i * 0.5) 
      for i in range(4)]
)
""").body[0].value
    
    def _create_thumbnail(self, filename: str) -> ast.Call:
        """Create thumbnail placeholder."""
        return ast.parse("""
Rectangle(
    width=16/9 * 2,
    height=2,
    color=PURPLE
)
""").body[0].value
    
    def _create_document(self, filename: str) -> ast.Call:
        """Create a document/paper representation."""
        return ast.parse("""
VGroup(
    Rectangle(width=2.5, height=3.5, color=WHITE, fill_opacity=0.8, stroke_color=BLACK),
    VGroup(*[Line(LEFT * 0.8, RIGHT * 0.8, color=GREY).shift(DOWN * i * 0.3) 
             for i in range(-3, 4)]).shift(UP * 0.2)
)
""").body[0].value
    
    def _create_arrow(self, filename: str) -> ast.Call:
        """Create an arrow or pointer."""
        return ast.parse("""
Arrow(LEFT, RIGHT, color=YELLOW)
""").body[0].value
    
    def _create_generic_shape(self, filename: str) -> ast.Call:
        """Create generic shape for unknown assets."""
        # Use filename to add context
        name = filename.split('/')[-1].split('.')[0][:10]  # First 10 chars
        return ast.parse(f"""
VGroup(
    Square(side_length=1, color=GREY),
    Text("{name}", font_size=16, color=WHITE).scale(0.5)
)
""").body[0].value
    
    def _apply_modifiers(self, base_node: ast.Call, modifiers: list) -> ast.Call:
        """Apply the original modifiers to the replacement node."""
        # For now, return the base node
        # In a full implementation, we'd chain the method calls
        return base_node
    
    def transform_code(self, code: str) -> Tuple[str, Dict[str, int]]:
        """Transform code by replacing asset calls."""
        try:
            tree = ast.parse(code)
            new_tree = self.visit(tree)
            
            stats = {
                'replaced': len(self.replacements),
                'failed': len(self.failed_replacements),
                'characters': sum(1 for r in self.replacements if self._determine_strategy(r) == 'character'),
                'buildings': sum(1 for r in self.replacements if self._determine_strategy(r) == 'building'),
                'symbols': sum(1 for r in self.replacements if self._determine_strategy(r) == 'math_symbol'),
                'other': len(self.replacements) - sum(1 for r in self.replacements if self._determine_strategy(r) in ['character', 'building', 'math_symbol'])
            }
            
            return ast.unparse(new_tree), stats
            
        except Exception as e:
            logger.error(f"Failed to transform assets: {e}")
            return code, {'error': str(e)}


def replace_assets_in_code(code: str) -> Tuple[str, Dict[str, int]]:
    """Main entry point for asset replacement."""
    replacer = AssetReplacer()
    return replacer.transform_code(code)