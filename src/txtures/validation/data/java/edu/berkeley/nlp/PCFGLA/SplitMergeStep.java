// From a saved Berkeley grammar, run the merge step exactly as GrammarTrainer does (merge
// weights, deltas, pairs, doTheMerges, and the lexicon-only E-step), or a split with
// randomness 0, and dump the results for checking the txtures trainer.
// Usage: SplitMergeStep merge|split TREES GRAMMAR.gr OUT
package edu.berkeley.nlp.PCFGLA;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.StateSet;
import edu.berkeley.nlp.util.Numberer;
import java.io.PrintWriter;
import java.util.*;

public class SplitMergeStep {
  public static void main(String[] args) throws Exception {
    ParserData pData = ParserData.Load(args[2]);
    Numberer.setNumberers(pData.getNumbs());
    Grammar maxGrammar = pData.getGrammar();
    SophisticatedLexicon maxLexicon = (SophisticatedLexicon) pData.getLexicon();
    Corpus corpus = new Corpus(args[1], Corpus.TreeBankType.SINGLEFILE, 1.0, false, -1, false, false);
    List<Tree<String>> trees = Corpus.binarizeAndFilterTrees(corpus.getTrainTrees(), 1, 0, 10000, Binarization.RIGHT, false, false);
    Numberer tagNumberer = Numberer.getGlobalNumberer("tags");
    StateSetTreeList train = new StateSetTreeList(trees, maxGrammar.numSubStates, false, tagNumberer);
    PrintWriter out = new PrintWriter(args[3] + ".extra", "UTF-8");
    if (args[0].equals("nodes")) {
      ArrayParser parser = new ArrayParser(maxGrammar, maxLexicon);
      int t = 0;
      for (Tree<StateSet> tree : train) {
        if (t++ >= Integer.parseInt(System.getProperty("trees", "300"))) break;
        parser.doInsideOutsideScores(tree, false, false);
        out.println("TREE " + (t - 1));
        nodes(out, tree);
      }
      out.close();
      return;
    }
    if (args[0].equals("merge")) {
      double[][] mergeWeights = GrammarMerger.computeMergeWeights(maxGrammar, maxLexicon, train);
      double[][][] deltas = GrammarMerger.computeDeltas(maxGrammar, maxLexicon, mergeWeights, train);
      boolean[][][] pairs = GrammarMerger.determineMergePairs(deltas, false, 0.5, maxGrammar);
      for (int s = 0; s < maxGrammar.numSubStates.length; s++) {
        out.println("MW " + s + OneStep.vec(mergeWeights[s], maxGrammar.numSubStates[s]));
        StringBuilder d = new StringBuilder("D " + s), p = new StringBuilder("P " + s);
        for (int k = 0; 2 * k + 1 < maxGrammar.numSubStates[s]; k++) {
          d.append(' ').append(deltas[s][2 * k][2 * k + 1]);
          p.append(' ').append(pairs[s][2 * k][2 * k + 1] ? 1 : 0);
        }
        out.println(d); out.println(p);
      }
      Grammar grammar = GrammarMerger.doTheMerges(maxGrammar, maxLexicon, pairs, mergeWeights);
      train = new StateSetTreeList(train, grammar.numSubStates, false);
      SophisticatedLexicon lexicon = new SophisticatedLexicon(grammar.numSubStates,
          SophisticatedLexicon.DEFAULT_SMOOTHING_CUTOFF, maxLexicon.getSmoothingParams(),
          maxLexicon.getSmoother(), maxLexicon.getPruningThreshold());
      double ll = GrammarTrainer.doOneEStep(grammar, maxLexicon, null, lexicon, train, true, 20);
      lexicon.optimize();
      splitTrees(out, grammar);
      out.close();
      OneStep.dump(args[3], grammar, lexicon, tagNumberer, ll);
    } else {
      CorpusStatistics stats = new CorpusStatistics(tagNumberer, train);
      int[] counts = stats.getSymbolCounts();
      Grammar grammar = maxGrammar.splitAllStates(0.0, counts, false, 0);
      Lexicon lexicon = maxLexicon.splitAllStates(counts, false, 0);
      splitTrees(out, grammar);
      out.close();
      OneStep.dump(args[3], grammar, (SophisticatedLexicon) lexicon, tagNumberer, Double.NaN);
    }
  }

  static void nodes(PrintWriter out, Tree<StateSet> tree) {
    if (tree.isLeaf()) return;
    StateSet l = tree.getLabel();
    StringBuilder b = new StringBuilder("N " + l.getState() + " " + l.getIScale() + " " + l.getOScale() + " I");
    for (int i = 0; i < l.numSubStates(); i++) b.append(' ').append(l.getIScore(i));
    b.append(" O");
    for (int i = 0; i < l.numSubStates(); i++) b.append(' ').append(l.getOScore(i));
    out.println(b);
    for (Tree<StateSet> c : tree.getChildren()) nodes(out, c);
  }

  static void splitTrees(PrintWriter out, Grammar g) {
    for (int s = 0; s < g.splitTrees.length; s++) out.println("ST " + s + " " + g.splitTrees[s]);
  }
}
