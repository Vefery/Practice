using System.IO;
using UnityEngine;

public class StatisticWriter : MonoBehaviour
{
    public int evals = 100;
    public int startSeed = 1;
    private int counter = 0;
    private Maze maze;
    private MazeAgent learnAgent;
    private DeterministicAgent deterministicAgent;

    private void Awake()
    {
        maze = FindFirstObjectByType<Maze>(FindObjectsInactive.Include);
        learnAgent = FindFirstObjectByType<MazeAgent>(FindObjectsInactive.Include);
        deterministicAgent = FindFirstObjectByType<DeterministicAgent>(FindObjectsInactive.Include);

    }
    private void Start()
    {
        deterministicAgent.gameObject.SetActive(false);
        learnAgent.gameObject.SetActive(true);
        maze.initDepth = 20;
        maze.seed = startSeed;
        maze.FinishEvent += HandleFinish;
    }
    private void HandleFinish()
    {
        if (counter >= evals)
        {
            string path = Application.dataPath + "/Stats/Determ.txt";
            using StreamWriter writer = new(path, true);
            writer.WriteLine(maze.initDepth + " " + deterministicAgent.steps);
        }
        else 
        {
            string path = Application.dataPath + "/Stats/Neuro.txt";
            using StreamWriter writer = new(path, true);
            writer.WriteLine(maze.initDepth + " " + learnAgent.steps);
        }
        counter++;
        maze.seed++;

        if (counter == evals)
        {
            Debug.Log("Neural done");
            maze.initDepth = 20;
            maze.solvedTimes = 0;
            maze.seed = startSeed;
            learnAgent.gameObject.SetActive(false);
            deterministicAgent.gameObject.SetActive(true);
        }
        else if (counter == evals * 2)
        {
            Debug.Log("Determ done");
            learnAgent.gameObject.SetActive(false);
            deterministicAgent.gameObject.SetActive(false);
        }
    }
}
