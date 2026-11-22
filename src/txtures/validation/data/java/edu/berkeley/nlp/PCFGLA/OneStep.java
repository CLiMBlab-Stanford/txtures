// Load a saved Berkeley grammar, run one EM iteration exactly as GrammarTrainer's loop does,
// and dump the grammar and lexicon state before and after, for checking the txtures trainer.
// Usage: OneStep TREES GRAMMAR.gr BEFORE.txt AFTER.txt
package edu.berkeley.nlp.PCFGLA;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.Numberer;
import java.io.PrintWriter;
import java.util.*;

public class OneStep {
  public static void main(String[] args) throws Exception {
    ParserData pData = ParserData.Load(args[1]);
    Numberer.setNumberers(pData.getNumbs());
    Grammar previousGrammar = pData.getGrammar();
    SophisticatedLexicon previousLexicon = (SophisticatedLexicon) pData.getLexicon();
    Corpus corpus = new Corpus(args[0], Corpus.TreeBankType.SINGLEFILE, 1.0, false, -1, false, false);
    List<Tree<String>> trees = Corpus.binarizeAndFilterTrees(corpus.getTrainTrees(), 1, 0, 10000, Binarization.RIGHT, false, false);
    Numberer tagNumberer = Numberer.getGlobalNumberer("tags");
    StateSetTreeList train = new StateSetTreeList(trees, previousGrammar.numSubStates, false, tagNumberer);
    dump(args[2], previousGrammar, previousLexicon, tagNumberer, Double.NaN);
    Grammar grammar = new Grammar(previousGrammar.numSubStates, previousGrammar.findClosedPaths,
        previousGrammar.smoother, previousGrammar, previousGrammar.threshold);
    SophisticatedLexicon lexicon = new SophisticatedLexicon(previousGrammar.numSubStates,
        SophisticatedLexicon.DEFAULT_SMOOTHING_CUTOFF, previousLexicon.getSmoothingParams(),
        previousLexicon.getSmoother(), previousLexicon.getPruningThreshold());
    double ll = GrammarTrainer.doOneEStep(previousGrammar, previousLexicon, grammar, lexicon, train, false, 20);
    lexicon.optimize();
    grammar.optimize(0);
    dump(args[3], grammar, lexicon, tagNumberer, ll);
  }

  static void smoother(PrintWriter out, String tag, Object smoother, short[] n) throws Exception {
    if (smoother == null || !smoother.getClass().getSimpleName().equals("SmoothAcrossParentBits")) return;
    java.lang.reflect.Field f = smoother.getClass().getDeclaredField("diffWeights");
    f.setAccessible(true);
    double[][][] d = (double[][][]) f.get(smoother);
    for (int s = 0; s < d.length; s++) {
      StringBuilder b = new StringBuilder(tag + " " + s);
      for (int i = 0; i < d[s].length; i++) b.append(vec(d[s][i], d[s][i].length));
      out.println(b);
    }
  }

  public static String vec(double[] v, int n) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) b.append(' ').append(v == null ? 0.0 : v[i]);
    return b.toString();
  }

  public static void dump(String path, Grammar g, SophisticatedLexicon lex, Numberer tags, double ll) throws Exception {
    PrintWriter out = new PrintWriter(path, "UTF-8");
    out.println("LL " + ll);
    short[] n = g.numSubStates;
    for (int s = 0; s < n.length; s++) out.println("TAG " + s + " " + tags.object(s) + " " + n[s]);
    for (int s = 0; s < n.length; s++) {
      for (BinaryRule r : g.splitRulesWithP(s)) {
        double[][][] sc = r.getScores2();
        StringBuilder b = new StringBuilder("B " + r.parentState + " " + r.leftChildState + " " + r.rightChildState);
        for (int l = 0; l < n[r.leftChildState]; l++)
          for (int c = 0; c < n[r.rightChildState]; c++) b.append(vec(sc[l][c], n[r.parentState]));
        out.println(b);
      }
      for (UnaryRule r : g.getUnaryRulesByParent(s)) {
        double[][] sc = r.getScores2();
        StringBuilder b = new StringBuilder("U " + r.parentState + " " + r.childState);
        for (int c = 0; c < n[r.childState]; c++) b.append(vec(sc[c], n[r.parentState]));
        out.println(b);
      }
    }
    for (int t = 0; t < n.length; t++) {
      if (lex.tagCounter[t] != null) out.println("T " + t + vec(lex.tagCounter[t], n[t]));
      if (lex.unseenTagCounter[t] != null) out.println("TU " + t + vec(lex.unseenTagCounter[t], n[t]));
      if (lex.wordToTagCounters[t] != null)
        for (Map.Entry<String, double[]> e : lex.wordToTagCounters[t].entrySet())
          out.println("L " + t + " " + e.getKey() + vec(e.getValue(), n[t]));
      if (lex.unseenWordToTagCounters[t] != null)
        for (Map.Entry<String, double[]> e : lex.unseenWordToTagCounters[t].entrySet())
          out.println("LU " + t + " " + e.getKey() + vec(e.getValue(), n[t]));
    }
    for (String w : lex.wordCounter.keySet()) out.println("W " + w + " " + lex.wordCounter.getCount(w));
    out.println("TOT " + lex.totalTokens + " " + lex.totalUnseenTokens);
    if (g.splitTrees != null)
      for (int s = 0; s < g.splitTrees.length; s++) out.println("ST " + s + " " + g.splitTrees[s]);
    smoother(out, "GS", g.smoother, n);
    smoother(out, "LS", lex.getSmoother(), n);
    out.close();
  }
}
