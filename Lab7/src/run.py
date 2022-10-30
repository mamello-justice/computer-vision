from classes import *

if __name__ == '__main__':
    puzzle = Puzzle(MATCH_IMGS)
    corner_piece = puzzle.pieces[3]

    # Start BFS by adding in the bottom left corner piece
    queue = []
    queue.append(corner_piece)
    corner_piece.insert()
    corner_piece.inserted = True

    while queue:
        source = queue.pop(0)

        for edge in source.edge_list:
            if edge is None:
                continue

            connected_edge = edge.connected_edge
            if connected_edge is None:
                continue

            parent_piece = connected_edge.parent_piece
            if parent_piece.inserted:
                continue

            queue.append(parent_piece)
            parent_piece.insert()
            parent_piece.inserted = True

    puzzle.display()
