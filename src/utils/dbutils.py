import sqlite3


class RayTracingDatabase:

    def __init__(self, dbfile):
        self.conn = sqlite3.connect(dbfile)

    def get_tx_positions(self, txid):
        cursor = self.conn.execute("SELECT x, y, z from tx where tx_set_id={0};".format(txid))
        tx_positions = []
        for row in cursor:
            tx_positions.append((row[0], row[1], row[2]))
        return tx_positions

    def get_rx_positions(self, rxid):
        cursor = self.conn.execute("SELECT x, y, z from rx where rx_set_id={0};".format(rxid))
        rx_positions = []
        for row in cursor:
            rx_positions.append((row[0], row[1], row[2]))
        return rx_positions
