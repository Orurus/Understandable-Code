"""
SimpLang Parser — Converts tokens into an AST with clear error messages.
"""

try:
    from .lexer import Token, TOKEN, SimpLangError
    from .ast_nodes import *
except ImportError:
    from lexer import Token, TOKEN, SimpLangError
    from ast_nodes import *


class ParserError(SimpLangError):
    """Parser-specific error."""
    pass


class Parser:
    """Recursive descent parser for SimpLang."""

    def __init__(self, tokens: list, source: str = "<unknown>"):
        self.tokens = tokens
        self.source = source
        self.pos = 0

    def error(self, message: str) -> ParserError:
        tok = self.peek()
        return ParserError(message, tok.line, tok.col, self.source)

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def check(self, *types: str) -> bool:
        """Check if the current token matches any of the given types."""
        return self.peek().type in types

    def expect(self, expected_type: str) -> Token:
        """Consume a token of expected_type or raise an error."""
        tok = self.peek()
        if tok.type == expected_type:
            return self.advance()
        raise self.error(f"Expected '{expected_type}' but got '{tok.value}' ({tok.type})")

    def expect_value(self, expected_value: str) -> Token:
        """Consume a token with expected string value."""
        tok = self.peek()
        if tok.value == expected_value:
            return self.advance()
        raise self.error(f"Expected '{expected_value}' but got '{tok.value}'")

    def mk_pos(self, tok: Token) -> Position:
        return Position(tok.line, tok.col, self.source)

    # ----------------------------------------------------------------
    # TOP-LEVEL: program = statement*
    # ----------------------------------------------------------------
    def parse(self) -> Program:
        """Parse entire program."""
        statements = []
        while not self.check(TOKEN['EOF']):
            stmt = self.parse_statement()
            if stmt is not None:
                statements.append(stmt)
        return Program(statements)

    # ----------------------------------------------------------------
    # STATEMENTS
    # ----------------------------------------------------------------
    def parse_statement(self):
        """Parse a single statement."""
        pos_tok = self.peek()

        # import name
        if self.check(TOKEN['IMPORT']):
            return self.parse_import()

        # fun name(...)
        if self.check(TOKEN['FUN']):
            return self.parse_fun_def()

        # ret expression
        if self.check(TOKEN['RET']):
            return self.parse_return()

        # say(...)
        if self.check(TOKEN['SAY']):
            return self.parse_say()

        # ask(...)
        if self.check(TOKEN['ASK']):
            return self.parse_ask()

        # if condition
        if self.check(TOKEN['IF']):
            return self.parse_if()

        # loop ...
        if self.check(TOKEN['LOOP']):
            return self.parse_loop()

        # end keyword (already handled at block level, but catch standalone)
        if self.check(TOKEN['END']):
            self.advance()
            return None  # end is a block terminator, not a real statement

        # expression (assignment, function call, etc.)
        return self.parse_expression_stmt()

    def parse_import(self):
        """import name or import /path/to/module.simp"""
        tok = self.advance()  # import
        start_line = tok.line

        tokens = []
        while self.pos < len(self.tokens):
            next_tok = self.peek()
            if next_tok.type == TOKEN['EOF'] or next_tok.line != start_line:
                break
            if next_tok.type in (TOKEN['IDENTIFIER'], TOKEN['DIV'], TOKEN['DOT']):
                tokens.append(self.advance())
            else:
                break

        if not tokens:
            raise self.error("Expected a module name or path after 'import'")

        name = ''.join(token.value for token in tokens)
        return ImportStmt(pos=self.mk_pos(tok), name=name)

    def parse_fun_def(self):
        """fun name(params) ... end"""
        tok = self.advance()  # fun
        name = self.expect(TOKEN['IDENTIFIER']).value
        self.expect(TOKEN['LPAREN'])
        params = []
        if not self.check(TOKEN['RPAREN']):
            params.append(self.expect(TOKEN['IDENTIFIER']).value)
            while self.check(TOKEN['COMMA']):
                self.advance()
                params.append(self.expect(TOKEN['IDENTIFIER']).value)
        self.expect(TOKEN['RPAREN'])
        body = self.parse_block(['end'])
        self.expect_value('end')
        return FunDef(pos=self.mk_pos(tok), name=name, params=params, body=body)

    def parse_return(self):
        """ret expression"""
        tok = self.advance()  # ret
        expr = self.parse_expression()
        return ReturnStmt(pos=self.mk_pos(tok), expr=expr)

    def parse_say(self):
        """say(expression)"""
        tok = self.advance()  # say
        self.expect(TOKEN['LPAREN'])
        expr = self.parse_expression()
        self.expect(TOKEN['RPAREN'])
        return SayStmt(pos=self.mk_pos(tok), expr=expr)

    def parse_ask(self):
        """ask(expression?)"""
        tok = self.advance()  # ask
        self.expect(TOKEN['LPAREN'])
        expr = None
        if not self.check(TOKEN['RPAREN']):
            expr = self.parse_expression()
        self.expect(TOKEN['RPAREN'])
        return AskStmt(pos=self.mk_pos(tok), expr=expr)

    def parse_if(self):
        """if condition body [else body] end"""
        tok = self.advance()  # if
        condition = self.parse_expression()
        body = self.parse_block(['else', 'end'])
        else_body = []
        if self.check(TOKEN['ELSE']):
            self.advance()  # else
            else_body = self.parse_block(['end'])
        self.expect_value('end')
        return IfStmt(pos=self.mk_pos(tok), condition=condition, body=body, else_body=else_body)

    def parse_loop(self):
        """loop while condition ... end  OR  loop var = start to end ... end  OR  loop var in iterable ... end"""
        tok = self.advance()  # loop

        # loop while
        if self.check(TOKEN['WHILE']):
            self.advance()
            condition = self.parse_expression()
            body = self.parse_block(['end'])
            self.expect_value('end')
            return WhileLoop(pos=self.mk_pos(tok), condition=condition, body=body)

        # Must be an identifier
        var_name = self.expect(TOKEN['IDENTIFIER']).value

        # loop var in iterable
        if self.check(TOKEN['IN']):
            self.advance()
            iterable = self.parse_expression()
            body = self.parse_block(['end'])
            self.expect_value('end')
            return ForInLoop(pos=self.mk_pos(tok), var_name=var_name, iterable=iterable, body=body)

        # loop var = start to end
        self.expect(TOKEN['ASSIGN'])
        start = self.parse_expression()
        self.expect_value('to')
        end = self.parse_expression()
        body = self.parse_block(['end'])
        self.expect_value('end')
        return ForRangeLoop(pos=self.mk_pos(tok), var_name=var_name, start=start, end=end, body=body)

    def parse_expression_stmt(self):
        """Parse an expression used as a statement (assignment, call, etc.)"""
        tok = self.peek()
        expr = self.parse_expression()

        if isinstance(expr, AssignStmt):
            return AssignStmt(pos=self.mk_pos(tok), name=getattr(expr.target, 'name', ''), target=expr.target, expr=expr.expr)

        return ExprStmt(pos=self.mk_pos(tok), expr=expr)

    def parse_block(self, terminators: list) -> list:
        """Parse statements until we hit one of the terminator keywords."""
        statements = []
        while self.pos < len(self.tokens):
            tok = self.peek()
            if tok.type == TOKEN['EOF']:
                raise self.error(f"Expected one of {terminators} but reached end of file")
            if tok.type in (TOKEN['END'], TOKEN['ELSE']) or tok.value in terminators:
                break
            stmt = self.parse_statement()
            if stmt is not None:
                statements.append(stmt)
        return statements

    # ----------------------------------------------------------------
    # EXPRESSIONS (PEMDAS: + - * / == != < > <= >= and or not)
    # ----------------------------------------------------------------
    def _is_assignable(self, node):
        return isinstance(node, (Identifier, Subscript))

    def parse_expression(self):
        """Lowest precedence: assignment, then and / or"""
        left = self.parse_comparison()

        if self.check(TOKEN['ASSIGN']) and self._is_assignable(left):
            self.advance()
            right = self.parse_expression()
            return AssignStmt(target=left, expr=right)

        while self.check(TOKEN['AND']) or self.check(TOKEN['OR']):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(left=left, op=op, right=right)
        return left

    def _parse_symbolic_comparison_phrase(self, left):
        """Parse English-style comparison phrases with optional joining words."""
        if self.check(TOKEN['IS']):
            self.advance()

        if self.check(TOKEN['NOT']):
            self.advance()
            if self.check(TOKEN['EQUAL']):
                self.advance()
            if self.check(TOKEN['TO']):
                self.advance()
            right = self.parse_term()
            return BinaryOp(left=left, op='!=', right=right)

        if not self.check(TOKEN['BIGGER'], TOKEN['SMALLER'], TOKEN['EQUAL']):
            return None

        op_token = self.advance().type

        if self.check(TOKEN['OR'], TOKEN['AND']) and self.peek(1).type == TOKEN['EQUAL']:
            self.advance()
            if self.check(TOKEN['EQUAL']):
                self.advance()
            if self.check(TOKEN['TO']):
                self.advance()
            right = self.parse_term()

            if op_token == TOKEN['BIGGER']:
                op = '>='
            elif op_token == TOKEN['SMALLER']:
                op = '<='
            else:
                op = '=='
            return BinaryOp(left=left, op=op, right=right)

        if self.check(TOKEN['THAN']):
            self.advance()
        elif op_token == TOKEN['EQUAL'] and self.check(TOKEN['TO']):
            self.advance()

        right = self.parse_term()

        if op_token == TOKEN['BIGGER']:
            op = '>'
        elif op_token == TOKEN['SMALLER']:
            op = '<'
        else:
            op = '=='
        return BinaryOp(left=left, op=op, right=right)

    def parse_comparison(self):
        """Parse comparison expressions using either symbol or English-style phrases."""
        left = self.parse_term()

        while True:
            if self.check(TOKEN['EQ']) or self.check(TOKEN['NEQ']) or self.check(TOKEN['LT']) or self.check(TOKEN['GT']) or self.check(TOKEN['LTE']) or self.check(TOKEN['GTE']):
                op = self.advance().value
                right = self.parse_term()
                left = BinaryOp(left=left, op=op, right=right)
                continue

            phrase = self._parse_symbolic_comparison_phrase(left)
            if phrase is not None:
                left = phrase
                continue

            break

        return left

    def parse_term(self):
        """+ -"""
        left = self.parse_factor()
        while self.check(TOKEN['PLUS']) or self.check(TOKEN['MINUS']):
            op = self.advance().value
            right = self.parse_factor()
            left = BinaryOp(left=left, op=op, right=right)
        return left

    def parse_factor(self):
        """* /"""
        left = self.parse_unary()
        while self.check(TOKEN['MUL']) or self.check(TOKEN['DIV']) or self.check(TOKEN['MOD']):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(left=left, op=op, right=right)
        return left

    def parse_unary(self):
        """not -"""
        if self.check(TOKEN['NOT']) or self.check(TOKEN['MINUS']):
            op = self.advance().value
            right = self.parse_unary()
            return UnaryOp(op=op, right=right)
        return self.parse_primary()

    def parse_primary(self):
        """Primary expressions: literals, identifiers, (expr), calls, subscripts, lists"""
        tok = self.peek()

        # Number
        if self.check(TOKEN['NUMBER']):
            self.advance()
            value = int(tok.value) if '.' not in tok.value else float(tok.value)
            node = NumberLiteral(value=value, pos=self.mk_pos(tok))
            return self.parse_postfix(node)

        # String
        if self.check(TOKEN['STRING']):
            self.advance()
            node = StringLiteral(value=tok.value, pos=self.mk_pos(tok))
            return self.parse_postfix(node)

        # Boolean / None
        if self.check(TOKEN['YEAH']):
            self.advance()
            node = BoolLiteral(value=True, pos=self.mk_pos(tok))
            return self.parse_postfix(node)
        if self.check(TOKEN['NAH']):
            self.advance()
            node = BoolLiteral(value=False, pos=self.mk_pos(tok))
            return self.parse_postfix(node)
        if self.check(TOKEN['NOPE']):
            self.advance()
            node = NoneLiteral(pos=self.mk_pos(tok))
            return self.parse_postfix(node)

        # Function-call keywords (say, ask can also be used in expressions)
        if self.check(TOKEN['SAY']):
            self.advance()
            self.expect(TOKEN['LPAREN'])
            arg = self.parse_expression()
            self.expect(TOKEN['RPAREN'])
            node = Call(func=Identifier(name='print'), args=[arg])
            return self.parse_postfix(node)
        if self.check(TOKEN['ASK']):
            self.advance()
            arg = None
            if self.check(TOKEN['LPAREN']):
                self.advance()
                if not self.check(TOKEN['RPAREN']):
                    arg = self.parse_expression()
                self.expect(TOKEN['RPAREN'])
            if arg:
                node = Call(func=Identifier(name='SimpLang_input'), args=[arg])
            else:
                node = Call(func=Identifier(name='SimpLang_input'), args=[])
            return self.parse_postfix(node)

        # Identifier
        if self.check(TOKEN['IDENTIFIER']):
            self.advance()
            node = Identifier(name=tok.value, pos=self.mk_pos(tok))
            return self.parse_postfix(node)

        # Type constructors: num(...), dec(...), txt(...)
        if self.check(TOKEN['NUM'], TOKEN['DEC'], TOKEN['TXT']):
            type_keyword = self.advance().value
            self.expect(TOKEN['LPAREN'])
            arg = self.parse_expression()
            self.expect(TOKEN['RPAREN'])
            # Convert to Python function call
            py_map = {'num': 'int', 'dec': 'float', 'txt': 'str'}
            node = Call(func=Identifier(name=py_map[type_keyword]), args=[arg])
            return self.parse_postfix(node)

        # (expr)
        if self.check(TOKEN['LPAREN']):
            self.advance()
            node = self.parse_expression()
            self.expect(TOKEN['RPAREN'])
            return self.parse_postfix(node)

        # List literal: [items]
        if self.check(TOKEN['LBRACKET']):
            return self.parse_list()

        raise self.error(f"Unexpected token '{tok.value}' ({tok.type})")

    def parse_postfix(self, node):
        """Handle postfix operations: calls and subscripts."""
        while True:
            # func(...)
            if self.check(TOKEN['LPAREN']):
                self.advance()
                args = []
                if not self.check(TOKEN['RPAREN']):
                    args.append(self.parse_expression())
                    while self.check(TOKEN['COMMA']):
                        self.advance()
                        args.append(self.parse_expression())
                self.expect(TOKEN['RPAREN'])
                node = Call(func=node, args=args)
                continue
            # expr[index]
            if self.check(TOKEN['LBRACKET']):
                self.advance()
                index = self.parse_expression()
                self.expect(TOKEN['RBRACKET'])
                node = Subscript(expr=node, index=index)
                continue
            # .member (for stdlib.xxx)
            if self.check(TOKEN['DOT']):
                self.advance()
                member = self.expect(TOKEN['IDENTIFIER']).value
                node = Call(func=Identifier(name='getattr'), args=[node, StringLiteral(value=member)])
                continue
            break
        return node

    def parse_list(self):
        """Parse a list literal: [item, item, ...]"""
        tok = self.advance()  # [
        items = []
        if not self.check(TOKEN['RBRACKET']):
            items.append(self.parse_expression())
            while self.check(TOKEN['COMMA']):
                self.advance()
                items.append(self.parse_expression())
        self.expect(TOKEN['RBRACKET'])
        return ListLiteral(items=items, pos=self.mk_pos(tok))