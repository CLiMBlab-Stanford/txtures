// Print the trees GrammarTrainer trains on, as ModelBlocks runs it:
// SINGLEFILE corpus, vertical markovization 1, horizontal 0, RIGHT binarization.
import edu.berkeley.nlp.PCFGLA.*;
import edu.berkeley.nlp.syntax.Tree;
import java.util.List;
public class DumpTrees {
  public static void main(String[] args) throws Exception {
    Corpus corpus = new Corpus(args[0], Corpus.TreeBankType.SINGLEFILE, 1.0, false, -1, false, false);
    List<Tree<String>> trees = Corpus.binarizeAndFilterTrees(corpus.getTrainTrees(), 1, 0, 10000, Binarization.RIGHT, false, false);
    for (Tree<String> t : trees) System.out.println(t.toString());
  }
}
