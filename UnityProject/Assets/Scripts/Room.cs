using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

public class Room : MonoBehaviour
{
    public int halfWidth = 10;
    public float neighbourChance = 0.25f;
    public Room left, right, up, down;
    private bool leftAllowed = true, rightAllowed = true, upAllowed = true, downAllowed = true;
    public int timesLeft = 0, timesRight = 0, timesUp = 0, timesDown = 0;
    public int timesVisited = 0;
    [System.NonSerialized]
    public int neighboursCount = 1;
    [System.NonSerialized]
    public List<Room> roomsList;
    public void Generate(Maze maze, Random.State state)
    {
        GenerationLoop(maze, state);
    }
    public void GenerateRooms(Maze maze, Room parentRoom, MazeDirection parentDirection, Random.State state)
    {
        switch (parentDirection)
        {
            case MazeDirection.left:
                left = parentRoom;
                leftAllowed = false;
                break;
            case MazeDirection.right:
                right = parentRoom;
                rightAllowed = false;
                break;
            case MazeDirection.up:
                up = parentRoom;
                upAllowed = false;
                break;
            case MazeDirection.down:
                down = parentRoom;
                downAllowed = false;
                break;
            default:
                break;
        }
        maze.rooms.Add(this);
        maze.depth--;
        GenerationLoop(maze, state);
        if (maze.depth <= 0 && !maze.goalSpawned)
        {
            maze.goalSpawned = true;
            maze.targetHolder = this;
            maze.target = Instantiate(maze.targetPrefab, transform.position, Quaternion.identity, this.transform).transform;
        }
    }
    private void GenerationLoop(Maze maze, Random.State state)
    {
        int roomsGenerated = 0;
        Random.state = state;
        while (roomsGenerated == 0 && maze.depth > 0 && (rightAllowed || leftAllowed || upAllowed || downAllowed))
        {
            leftAllowed = maze.CheckPosition(transform.position + new Vector3(-halfWidth, 0, 0)) && left == null;
            if (Random.Range(0f, 1f) <= neighbourChance && leftAllowed)
            {
                left = Instantiate(maze.roomPrefab, transform.position + new Vector3(-halfWidth, 0, 0), Quaternion.identity, transform).GetComponent<Room>();
                maze.tilePositions.Add(transform.position + new Vector3(-halfWidth, 0, 0));
                maze.roomsCreateActions.Add(() => left.GenerateRooms(maze, this, MazeDirection.right, Random.state));
                roomsGenerated++;
                neighboursCount++;
            }
            rightAllowed = maze.CheckPosition(transform.position + new Vector3(halfWidth, 0, 0)) && right == null;
            if (Random.Range(0f, 1f) <= neighbourChance && rightAllowed)
            {
                right = Instantiate(maze.roomPrefab, transform.position + new Vector3(halfWidth, 0, 0), Quaternion.identity, transform).GetComponent<Room>();
                maze.tilePositions.Add(transform.position + new Vector3(halfWidth, 0, 0));
                maze.roomsCreateActions.Add(() => right.GenerateRooms(maze, this, MazeDirection.left, Random.state));
                roomsGenerated++;
                neighboursCount++;
            }
            upAllowed = maze.CheckPosition(transform.position + new Vector3(0, 0, halfWidth)) && up == null;
            if (Random.Range(0f, 1f) <= neighbourChance && upAllowed)
            {
                up = Instantiate(maze.roomPrefab, transform.position + new Vector3(0, 0, halfWidth), Quaternion.identity, transform).GetComponent<Room>();
                maze.tilePositions.Add(transform.position + new Vector3(0, 0, halfWidth));
                maze.roomsCreateActions.Add(() => up.GenerateRooms(maze, this, MazeDirection.down, Random.state));
                roomsGenerated++;
                neighboursCount++;
            }
            downAllowed = maze.CheckPosition(transform.position + new Vector3(0, 0, -halfWidth)) && down == null;
            if (Random.Range(0f, 1f) <= neighbourChance && downAllowed)
            {
                down = Instantiate(maze.roomPrefab, transform.position + new Vector3(0, 0, -halfWidth), Quaternion.identity, transform).GetComponent<Room>();
                maze.tilePositions.Add(transform.position + new Vector3(0, 0, -halfWidth));
                maze.roomsCreateActions.Add(() => down.GenerateRooms(maze, this, MazeDirection.up, Random.state));
                roomsGenerated++;
                neighboursCount++;
            }
        }

        if (left == null)
        {
            Instantiate(maze.wallPrefab, transform.position, Quaternion.identity, transform);
            timesLeft = 100;
        }
        if (right == null)
        {
            Instantiate(maze.wallPrefab, transform.position, Quaternion.Euler(0, 180, 0), transform);
            timesRight = 100;
        }
        if (up == null)
        {
            timesUp = 100;
            Instantiate(maze.wallPrefab, transform.position, Quaternion.Euler(0, 90, 0), transform);
        }
        if (down == null)
        {
            timesDown = 100;
            Instantiate(maze.wallPrefab, transform.position, Quaternion.Euler(0, -90, 0), transform);
        }

        roomsList = new List<Room> { right, down, left, up };
    }
}
