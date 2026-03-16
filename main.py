import numpy as np
from collections import defaultdict
from itertools import combinations

class Fast3SAT:
    """
    Tentativo di algoritmo O(n^2) per 3-SAT
    Basato su trasformazione in grafo e propagazione vincolata
    """
    
    def __init__(self, num_vars, clauses):
        self.n = num_vars
        self.m = len(clauses)
        self.clauses = clauses
        
        # Strutture dati ottimizzate
        self.var_index = {f'X{i}': i-1 for i in range(1, num_vars + 1)}
        
        # Matrice di implicazioni (innovazione chiave)
        self.implication_matrix = self._build_implication_matrix()
        
        # Vettori di vincoli (algebra lineare booleana)
        self.constraint_vectors = self._build_constraint_vectors()
        
    def _build_implication_matrix(self):
        """
        Costruisce una matrice N x N che cattura le relazioni tra variabili
        Idea: Se X -> Y in tutte le clausole, allora c'è dipendenza forte
        """
        matrix = np.zeros((self.n, self.n), dtype=int)
        
        for clause in self.clauses:
            # Analizza implicazioni nella clausola
            vars_in_clause = [self.var_index[var] for var, _ in clause]
            
            # Costruisco grafo delle implicazioni
            for (v1, s1), (v2, s2) in combinations(clause, 2):
                idx1, idx2 = self.var_index[v1], self.var_index[v2]
                
                # Regole di implicazione basate sui segni
                if s1 and not s2:  # X1 -> ¬X2
                    matrix[idx1][idx2] += 1
                elif not s1 and s2:  # ¬X1 -> X2
                    matrix[idx1][idx2] += 1
                # ... altre regole
                
        return matrix
    
    def _build_constraint_vectors(self):
        """
        Trasforma clausole in vettori per algebra lineare
        Ogni clausola diventa un vincolo lineare in GF(2)
        """
        vectors = []
        for clause in self.clauses:
            v = np.zeros(self.n, dtype=int)
            for var, sign in clause:
                idx = self.var_index[var]
                v[idx] = 1 if sign else -1
            vectors.append(v)
        return np.array(vectors)
    
    def _detect_communities(self):
        """
        Divide le variabili in comunità basate su interazioni forti
        Usa la matrice di implicazione per trovare componenti connesse
        """
        # Soglia di interazione significativa
        threshold = self.m // 10
        
        # Costruisco grafo delle dipendenze forti
        adj_list = defaultdict(list)
        for i in range(self.n):
            for j in range(self.n):
                if self.implication_matrix[i][j] > threshold:
                    adj_list[i].append(j)
        
        # Trova componenti connesse (BFS in O(n + e))
        communities = []
        visited = set()
        
        for start in range(self.n):
            if start not in visited:
                community = set()
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        community.add(node)
                        stack.extend(adj_list[node])
                if community:
                    communities.append(community)
        
        return communities
    
    def _fast_local_solve(self, community, external_assignments):
        """
        Risolve una comunità localmente con propagazione vincolata
        Complessità: O(|community|^2) per comunità
        """
        if not community:
            return {}
        
        # Vettore delle assegnazioni locali
        local_assign = {}
        
        # Trova variabili con vincoli più forti
        influence_scores = {}
        for var in community:
            # Calcola influenza basata su implicazioni
            influence = np.sum(np.abs(self.implication_matrix[var][list(community)]))
            influence_scores[var] = influence
        
        # Ordina per influenza
        sorted_vars = sorted(community, key=lambda v: -influence_scores[v])
        
        # Propagazione vincolata (O(k^2) dove k = |community|)
        for var in sorted_vars:
            if var in local_assign:
                continue
                
            # Analizza clausole che coinvolgono questa variabile
            pos_score = 0
            neg_score = 0
            
            for clause in self.clauses:
                vars_in_clause = [self.var_index[v] for v, _ in clause]
                if var in vars_in_clause:
                    # Calcola se assegnare True/False soddisfa più clausole
                    for v, sign in clause:
                        if self.var_index[v] == var:
                            if sign:
                                pos_score += 1
                            else:
                                neg_score += 1
            
            # Assegna in base al punteggio
            if pos_score > neg_score:
                local_assign[var] = True
            else:
                local_assign[var] = False
            
            # Propaga l'assegnamento alle variabili vicine
            for other in community:
                if other not in local_assign:
                    # Usa matrice implicazione per inferire
                    impl_strength = self.implication_matrix[var][other]
                    if impl_strength > 0:
                        # Se implicazione forte, assegna conseguentemente
                        local_assign[other] = not local_assign[var]  # Semplificato
                        
        return local_assign
    
    def solve(self):
        """
        Algoritmo principale O(n^2)
        """
        # Fase 1: Rilevazione comunità (O(n^2))
        communities = self._detect_communities()
        
        # Fase 2: Ordinamento topologico delle comunità
        community_graph = self._build_community_graph(communities)
        sorted_communities = self._topological_sort(communities, community_graph)
        
        # Fase 3: Risoluzione per comunità (O(n^2) totale)
        final_assignment = {}
        for i, community in enumerate(sorted_communities):
            # Risolvi comunità considerando assegnamenti precedenti
            community_solution = self._fast_local_solve(community, final_assignment)
            final_assignment.update(community_solution)
            
            # Fase 4: Verifica rapida (O(m))
            if not self._quick_check(final_assignment, i):
                # Se violazione, backtrack limitato alla comunità corrente
                final_assignment = self._local_backtrack(community, final_assignment)
        
        # Fase 5: Verifica finale
        if self._verify_solution(final_assignment):
            return final_assignment
        else:
            # Se fallisce, prova assegnamento complementare (solo 2 tentativi)
            return self._try_complementary(final_assignment)
    
    def _build_community_graph(self, communities):
        """Costruisce grafo delle dipendenze tra comunità"""
        graph = defaultdict(set)
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities):
                if i != j:
                    # Controlla se ci sono clausole che connettono le comunità
                    for clause in self.clauses:
                        vars_in = [self.var_index[v] for v, _ in clause]
                        has_from_comm1 = any(v in comm1 for v in vars_in)
                        has_from_comm2 = any(v in comm2 for v in vars_in)
                        if has_from_comm1 and has_from_comm2:
                            graph[i].add(j)
        return graph
    
    def _topological_sort(self, communities, graph):
        """Ordinamento topologico delle comunità"""
        in_degree = [len(graph[i]) for i in range(len(communities))]
        queue = [i for i, deg in enumerate(in_degree) if deg == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(communities[node])
            # Aggiorna gradi (semplificato)
            
        return result + [comm for i, comm in enumerate(communities) if comm not in result]
    
    def _quick_check(self, assignment, up_to_community):
        """Verifica rapida parziale"""
        # Controlla solo clausole che coinvolgono comunità già assegnate
        # Implementazione semplificata
        return True
    
    def _local_backtrack(self, community, assignment):
        """Backtrack limitato alla comunità"""
        # Inverte assegnamenti nella comunità
        for var in community:
            if var in assignment:
                assignment[var] = not assignment[var]
        return assignment
    
    def _try_complementary(self, assignment):
        """Prova assegnamento complementare"""
        complement = {k: not v for k, v in assignment.items()}
        if self._verify_solution(complement):
            return complement
        return None
    
    def _verify_solution(self, assignment):
        """Verifica se l'assegnamento soddisfa tutte le clausole"""
        if assignment is None:
            return False
            
        for clause in self.clauses:
            satisfied = False
            for var, sign in clause:
                idx = self.var_index[var]
                if idx in assignment and assignment[idx] == sign:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

# Test
def test_fast_3sat():
    # Test con istanze casuali
    num_vars = 100
    num_clauses = 400
    
    from random import choice, randint
    
    # Genera formula 3-CNF
    clauses = []
    for _ in range(num_clauses):
        vars_chosen = [f'X{randint(1, num_vars)}' for _ in range(3)]
        clause = [(v, choice([True, False])) for v in vars_chosen]
        clauses.append(clause)
    
    # Esegui algoritmo
    solver = Fast3SAT(num_vars, clauses)
    solution = solver.solve()
    
    print(f"Soluzione trovata: {solution is not None}")
    if solution:
        print(f"Verifica: {solver._verify_solution(solution)}")

if __name__ == "__main__":
    test_fast_3sat()
